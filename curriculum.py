"""
Implementation of DDPG - Deep Deterministic Policy Gradient https://github.com/pemami4911/deep-rl
Modified by Coac for BiCNet implementation https://github.com/Coac/CommNet-BiCnet
"""
import argparse
import pprint as pp
from datetime import datetime

import numpy as np
import tensorflow as tf
# from comm_net import CommNet
#from bicnet import BiCNet as CommNet
#from guessing_sum_env import *
from replay_buffer import ReplayBuffer
import traci
import matplotlib.pyplot as plt
import math
import os
import sys
import optparse


HIDDEN_VECTOR_LEN = 1
NUM_AGENTS =8
VECTOR_OBS_LEN = 3
Platoonsize=NUM_AGENTS
OUTPUT_LEN = 1

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable ")


try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

class CommNet:
    @staticmethod
    def base_build_network(observation):
        encoded = CommNet.shared_dense_layer("encoder", observation, HIDDEN_VECTOR_LEN)

        hidden_agents = tf.unstack(encoded, NUM_AGENTS, 1)

        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_VECTOR_LEN, forget_bias=1.0, name="lstm_fw_cell")
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_VECTOR_LEN, forget_bias=1.0, name="lstm_bw_cell")
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, hidden_agents, dtype=tf.float32)
        with tf.variable_scope("bidirectional_rnn", reuse=tf.AUTO_REUSE):
            tf.summary.histogram("lstm_fw_cell/kernel", tf.get_variable("fw/lstm_fw_cell/kernel"))
            tf.summary.histogram("lstm_bw_cell/kernel", tf.get_variable("bw/lstm_bw_cell/kernel"))

        outputs = tf.stack(outputs, 1)
        return outputs

    @staticmethod
    def actor_build_network(name, observation):
        with tf.variable_scope(name):
            outputs = CommNet.base_build_network(observation)
            return CommNet.shared_dense_layer("output_layer", outputs, OUTPUT_LEN)


    @staticmethod
    def shared_dense_layer(name, observation, output_len):
        H = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for j in range(NUM_AGENTS):
                agent_obs = observation[:, j]
                agent_encoded = tf.layers.dense(agent_obs, output_len, name="dense")
                tf.summary.histogram(name + "/dense/kernel", tf.get_variable("dense/kernel"))
                H.append(agent_encoded)
            H = tf.stack(H, 1)
        return H

    @staticmethod
    def critic_build_network(name, observation, action):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            outputs = CommNet.base_build_network(tf.concat([observation, action], 2))
            outputs = CommNet.shared_dense_layer("output_layer", outputs, 1)
            return outputs
        
class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = tf.Session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.inputs, self.out = self.create_actor_network("actor_network")
        self.network_params = tf.trainable_variables()

        self.target_inputs, self.target_out = self.create_actor_network("target_actor_network")
        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        with tf.name_scope("actor_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.action_gradient = tf.placeholder(tf.float32, (NUM_AGENTS, None, NUM_AGENTS, OUTPUT_LEN), name="action_gradient")


        with tf.name_scope("actor_gradients"):
            grads = []
            for i in range(NUM_AGENTS):
                for j in range(NUM_AGENTS):
                    grads.append(tf.gradients(self.out[:, j], self.network_params, -self.action_gradient[j][:, i]))
            grads = np.array(grads)
            self.unnormalized_actor_gradients = [tf.reduce_sum(list(grads[:, i]), axis=0) for i in range(len(self.network_params))]
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimize.apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="actor_inputs")
        out = CommNet.actor_build_network(name, inputs)
        return inputs, out

    def train(self, inputs, action_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: action_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = tf.Session()
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.inputs, self.action, self.out = self.create_critic_network("critic_network")
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network("target_critic_network")
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        with tf.name_scope("critic_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                      + tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, (None, NUM_AGENTS, 1), name="predicted_q_value")

        M = tf.to_float(tf.shape(self.out)[0])
        # Li = (Yi - Qi)^2
        # L = Sum(Li)
        self.loss = tf.squeeze(1.0/M * tf.reduce_sum(tf.reduce_sum(tf.square(self.predicted_q_value - self.out), axis=1), axis=0), name="critic_loss")

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # self.action_grads = tf.gradients(self.out, self.action, name="action_grads")
        self.action_grads = [tf.gradients(self.out[:, i], self.action) for i in range(NUM_AGENTS)]
        self.action_grads = tf.stack(tf.squeeze(self.action_grads, 1))
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN), name="critic_action")

        out = CommNet.critic_build_network(name, inputs, action)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0., name="episode_reward")
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0., name="episode_ave_max_q")
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    loss = tf.Variable(0., name="critic_loss")
    tf.summary.scalar("Critic_loss", loss)

    summary_vars = [episode_reward, episode_ave_max_q, loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess,  args, actor, critic):
    plt.ion() #开启interactive mode
    speedmode=6
    madr=1.4
    gapvector=[0]*16
    totalreward=[]
    
    
    le=10000
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    leading=[]
    
    
    
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'] +  " actor_lr" + str(args['actor_lr']) + " critic_lr" + str(args["critic_lr"]), sess.graph)

    actor.update_target_network()
    critic.update_target_network()

    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(1200):
#        print(i)
        zongreward=0
        locationplot=[]
        speedplot=[]
        timeplot=[]
        traci.start([sumoBinary,"-c","hello.sumocfg"])
#        print('shenme')
        locationplot=[]
        speedplot=[]
        
        timeplot=[]
        done=0
        chusudu=14
        for i in range(0,40):
            leading.append(0)
        for i in range(40,70):
            leading.append(-1)
        for i in range(70,200):
            leading.append(1)
            
        for step in range(100):
            exist_list=traci.vehicle.getIDList()
            if len(exist_list)>0:
                traci.vehicle.setSpeed(exist_list[0],chusudu)
            traci.simulationStep()
        gapvector=[2*chusudu]*16
#        print(gapvector)
        traci.vehicle.moveTo('a','L4_0',le)
        traci.vehicle.moveTo('b.0','L4_0',le-gapvector[0])
        traci.vehicle.moveTo('b.1','L4_0',le-sum(gapvector[:2]))
        traci.vehicle.moveTo('b.2','L4_0',le-sum(gapvector[:3]))
        traci.vehicle.moveTo('b.3','L4_0',le-sum(gapvector[:4]))
        traci.vehicle.moveTo('b.4','L4_0',le-sum(gapvector[:5]))
        traci.vehicle.moveTo('b.5','L4_0',le-sum(gapvector[:6]))
        traci.vehicle.moveTo('b.6','L4_0',le-sum(gapvector[:7]))
        traci.vehicle.moveTo('b.7','L4_0',le-sum(gapvector[:8]))
        traci.vehicle.moveTo('c.0','L4_0',le-sum(gapvector[:9]))
        traci.vehicle.moveTo('c.1','L4_0',le-sum(gapvector[:10]))
        traci.vehicle.moveTo('c.2','L4_0',le-sum(gapvector[:11]))
        traci.vehicle.moveTo('c.3','L4_0',le-sum(gapvector[:12]))
        traci.vehicle.moveTo('c.4','L4_0',le-sum(gapvector[:13]))
        traci.vehicle.moveTo('c.5','L4_0',le-sum(gapvector[:14]))
        traci.vehicle.moveTo('c.6','L4_0',le-sum(gapvector[:15]))
        traci.vehicle.moveTo('c.7','L4_0',le-sum(gapvector[:16]))
        traci.simulationStep()
        chushiweizhi=[]
        exist_list=traci.vehicle.getIDList()
        for xx in exist_list:
            chushiweizhi.append(traci.vehicle.getPosition(xx)[0])
        
        
        touche=leading

        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):
#            pjz=0
            
            initialsp=[]
            state2=[]
            state=[]
            reward=[]
#            print()
            xiayimiaosudu=np.clip(traci.vehicle.getSpeed(exist_list[0])+touche[j],0,chusudu)
            traci.vehicle.setSpeed(exist_list[0],xiayimiaosudu)
            for xx in exist_list:
                traci.vehicle.setSpeedMode(xx,speedmode)
                initialsp.append(traci.vehicle.getSpeed(xx))
                locationplot.append(traci.vehicle.getPosition(xx)[0]/1000)
                speedplot.append(traci.vehicle.getSpeed(xx))
                timeplot.append(j)

            for mm in range(1,NUM_AGENTS+1):
#                touchea=exist_list[0]
                ziji=exist_list[mm]
                qianche=exist_list[mm-1]
                gap=traci.vehicle.getLeader(ziji)[1]
                zhuangtai1=(traci.vehicle.getSpeed(qianche)-traci.vehicle.getSpeed(ziji))/10
                zhuangtai2=(traci.vehicle.getSpeed(ziji)-16)/16
                zhuangtai3=(math.sqrt(max(gap,0))-20)/20
                state.append([zhuangtai1,zhuangtai2,zhuangtai3])
                

            action = actor.predict([state])[0]
            chaoguo=[0]*NUM_AGENTS
            for mm in range(1,NUM_AGENTS+1):
                ziji=exist_list[mm]
                qianche=exist_list[mm-1]
                zijisudu=traci.vehicle.getSpeed(ziji)
                qianchesudu=traci.vehicle.getSpeed(qianche)
                gapa=traci.vehicle.getLeader(ziji)[1]
                if qianchesudu-3<zijisudu:
                    gap=gapa-5-zijisudu+max(qianchesudu-3,0)                        
                    if gap<0:
                        amax=-3
#                        print(gap)
                    else:
#                        amax=math.sqrt(madr*gap)+sp[i]-sp[i+1]-3
                        amax=min(gap/3,math.sqrt(madr*gap))+qianchesudu-zijisudu-3
                        amax=np.clip(amax,-3,3)
                else:
                    amax=3                
#                ac=np.clip(action[mm-1][0]/10,-3,3)
#                if pjz==0:
#                    ave=sum(action)/NUM_AGENTS
#                    pjz=1
                ac=np.clip(action[mm-1][0]/10,-3,3)
#                print(j,ave,action,ac)
                if ac>amax:
                    chaoguo[mm-1]=1
#                print(action[mm-1][0])
#                print(j,mm,ac,amax)
                nextspeed=traci.vehicle.getSpeed(exist_list[mm])+min(amax,ac)
#                nextspeed=traci.vehicle.getSpeed(exist_list[mm])+ac
#                print(action[mm-1][0])
                traci.vehicle.setSpeed(exist_list[mm],nextspeed)
            traci.simulationStep()
#            for i in NUM_AGENTS+1):
#                if i>0 and (po[i]>po[i-1]-5 or po[i]<-10000):
#                    chongtu[i-1]=1 
            chongtu=[0]*NUM_AGENTS
#            print(j)
            for mm in range(1,NUM_AGENTS+1):
                ziji=exist_list[mm]
                qianche=exist_list[mm-1]
#                print(traci.vehicle.getPosition(ziji)[0])
                if traci.vehicle.getPosition(ziji)[0]<-10000:
                    chongtu[mm-1]=1
                re=min((traci.vehicle.getAcceleration(ziji))**2/9,1)
#                print(mm-1,traci.vehicle.getAcceleration(ziji),re)
                if chongtu[mm-1]==0:
                    gap=traci.vehicle.getLeader(ziji)[1]
                else:
                    gap=0
                if gap>100:
                    re+=gap/100
#                print(mm-1,gap,re)
                if chaoguo[mm-1]==1:
                    re+=1
                if chongtu[mm-1]==1:
                    re+=5
#                    print('chaoguo'W)
#                print(mm-1,chaoguo[mm-1],re)
                    
                reward.append([1-re])
                done=True
            state2=None
            
            replay_buffer.add(state, action, reward, done, state2)
#            print(reward)

            if replay_buffer.size() > int(args['minibatch_size']) or sum(chongtu)>0:
                
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))
#                print(j)
#                print(chongtu)
                if j%33==32:
                    predicted_q_value, _, loss = critic.train(s_batch, a_batch,
                                                              np.reshape(r_batch, (32, NUM_AGENTS, 1)))
                else:
                    predicted_q_value, _, loss = critic.train(s_batch, a_batch,
                                                              np.reshape(r_batch, (j%33+1, NUM_AGENTS, 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads)

                actor.update_target_network()
                critic.update_target_network()
#                print('xunlianle')

                replay_buffer.clear()

                # Log
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: np.mean(r_batch),
                    summary_vars[1]: ep_ave_max_q / float(j + 1),
                    summary_vars[2]: loss
                })

                writer.add_summary(summary_str, i)
                writer.flush()
#                print(j,reward,r_batch,np.mean(r_batch))
                
                state=[]
                reward=[]

#                print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format(np.mean(r_batch),
#                                                                               i, (ep_ave_max_q / float(j + 1))))
                zongreward+=np.mean(r_batch)
                print(j,action,chaoguo)
            if sum(chongtu)>0:
                print(traci.vehicle.getIDCount())
                print('zhuangle22222222222222222222222222')
                replay_buffer.clear()
                traci.close()
                sys.stdout.flush()
#                bre=1
                break

        replay_buffer.clear()
        traci.close()
        sys.stdout.flush()
#        print(ave)
#            if state2!=None:
#                print(state,action,reward,state2)
#        print(totalreward,zongreward)
        print(j,zongreward/9-1)
        if j>180:
            totalreward.append(zongreward/9-1)
        plt.ion()
        plt.figure(i*2-1)
        plt.plot(np.arange(len(totalreward)), totalreward)
        plt.xlabel('Episode');plt.ylabel('Episode reward')
        plt.draw()
        plt.pause(1)
        plt.close()#越大越好
        
        plt.ion()
        plt.figure(i*2)
        plt.scatter(timeplot,locationplot,c=speedplot,s=10,alpha=0.3)
        plt.colorbar()
        plt.xlabel('Time (s)');plt.ylabel('Location (km)')
        plt.grid(True)
        plt.show()
        
    M8=np.mat(totalreward)
    np.savetxt("M8.csv", M8, delimiter=',')
    



def main(args=None):
    args = parse_arg(args or None)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    print('kaishi')
    with tf.Session(config=config) as sess:
#        env = GuessingSumEnv(NUM_AGENTS)
#        env.seed(0)

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
#        env.seed(int(args['random_seed']))

        state_dim = (NUM_AGENTS, VECTOR_OBS_LEN)
        action_dim = (NUM_AGENTS, OUTPUT_LEN)
        print('kaishi1')

        actor = ActorNetwork(sess, state_dim, action_dim,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        

        train(sess, args, actor, critic)
        
        Actorsave_path=actor.saver.save(actor.sess,"actormodel/model.ckpt")
        Criticsave_path=critic.saver.save(critic.sess,"criticmodel/model.ckpt")
        print("save to path:",Actorsave_path,Criticsave_path)


def parse_arg(args=None):
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.1)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.1)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.95)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=10000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=32)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=300)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=200)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default="summaries/" + datetime.now().strftime('%d-%m-%y %H%M'))

    if args is not None:
        args = vars(parser.parse_args(args))
    else:
        args = vars(parser.parse_args())

    pp.pprint(args)

    return args


if __name__ == '__main__':

    main()
    
