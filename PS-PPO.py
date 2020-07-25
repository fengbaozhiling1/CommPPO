# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:54:41 2020

@author: Think
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:55:39 2020

@author: Think
"""

import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import optparse
import traci
import math
#import pandas as pd


# SUMO软件的调用代码
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable ")


# 从SUMO列表中调用Python的Module
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary  # noqa
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")


# SUMO软件调用需要的函数
def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options

ave=20
Platoonsize=16
EP_MAX = 2000
EP_LEN = 300
GAMMA = 0.95
A_LR = 0.00001
C_LR = 0.00002
BATCH = 64
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 4, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization
vfree=33


class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        
        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
#                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                ratio=tf.divide(pi.prob(self.tfa), tf.maximum(oldpi.prob(self.tfa), 1e-5))
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
            

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def update(self, s, a, r):
#        with tf.Session():
#            print(self.closs.eval())
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
#        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

#        print('update actor')
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

#        print( 'update critic')
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]
    

def calcstate(che,po,sp,acc,kexing):#自己速度是sp[che+1]
#    print(che,acc,acc[che])
    s=[0]*4
#    s[0]=(sp[0]-sp[che+1])/10
    s[0]=(sp[che]-sp[che+1])/10
    s[1]=(sp[che+1]-16)/16
    s[2]=(math.sqrt(max(po[che]-po[che+1]-5,0))-20)/20
    s[3]=(che-8)/16

#    s[4]=acc[0]/3

    return s

#def calcreward(che,po,sp,acc,chongtu,chaoguo):#自己速度是sp[che+1]
#
#    v=sp[che+1]
#    a=acc[che+1]
#
#    headway=(po[che]-po[che+1])
#    if a+0.5*Cd*rouair*Af*v*v/m+miu*g>=0:
#        fv=0.1569+0.0245*v-0.0007415*v*v+0.00005975*v**3+a*(0.07224+0.09681*v+0.001075*v*v)
#    else:
#        fv=0
#    energy=fv/(v+0.1)
#    
#    if headway>100:
#        energy+=headway/100*3.7362
#        if che<2:
#            print(che,headway)
#
#    if chongtu==1 or chaoguo==1:
#        if che<2:
#            print(che,chaoguo)
#        energy+=3.7362
#    energy*=100
#    return -energy
def calcreward(che,po,sp,acc,chongtu,chaoguo):#自己速度是sp[che+1]
    headway=(po[che]-po[che+1])
    a=acc[che+1]
    energy=a**2/9
    if headway>100:
        energy+=headway/100
    if chongtu==1 or chaoguo==1:
        energy+=1
    energy*=100
    return -energy
if __name__ == "__main__":
    plt.ion() #开启interactive mode
    m=1200
    Af=2.5
    Cd=0.32
    rouair=1.184
    g=9.8
    speedmode=6
    miu=0.015
    con=[]
    ppo = PPO()
    all_ep_r = []
    Position=np.ones(Platoonsize+1)
    Speed=np.ones(Platoonsize+1)
    Acc=np.ones(Platoonsize+1)
    zuihouweizhi=[]
    initialpo=[0]*Platoonsize
    juli=[]
    xunlian=300
    madr=1.4
    gapvector=[0]*16
    rewardvector=[]
    
    le=10000
    options = get_options()
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')
    leading=[]
    
    
    for ep in range(xunlian*4):
        traci.start([sumoBinary,"-c","hello.sumocfg"])

        locationplot=[]
        speedplot=[]
        timeplot=[]
        done=0
        consumption=0
        distance=0
        
#        print(initialpo)   
        buffer_s, buffer_a, buffer_r = [], [], []
        
        s0=[]
        s1=[]
        s2=[]
        s3=[]
        s4=[]
        s5=[]
        s6=[]
        s7=[]
        s8=[]
        s9=[]
        s10=[]
        s11=[]
        s12=[]
        s13=[]
        s14=[]
        s15=[]
        
        
        r0=[]
        r1=[]
        r2=[]
        r3=[]
        r4=[]
        r5=[]
        r6=[]
        r7=[]
        r8=[]
        r9=[]
        r10=[]
        r11=[]
        r12=[]
        r13=[]
        r14=[]
        r15=[]
        
        
        a0=[]
        a1=[]
        a2=[]
        a3=[]
        a4=[]
        a5=[]
        a6=[]
        a7=[]
        a8=[]
        a9=[]
        a10=[]
        a11=[]
        a12=[]
        a13=[]
        a14=[]
        a15=[]
        
        s_0=[]
        s_1=[]
        s_2=[]
        s_3=[]
        s_4=[]
        s_5=[]
        s_6=[]
        s_7=[]
        s_8=[]
        s_9=[]
        s_10=[]
        s_11=[]
        s_12=[]
        s_13=[]
        s_14=[]
        s_15=[]
        
        ep_r0=0
        ep_r1=0
        ep_r2=0
        ep_r3=0
        ep_r4=0
        ep_r5=0
        ep_r6=0
        ep_r7=0
        ep_r8=0
        ep_r9=0
        ep_r10=0
        ep_r11=0
        ep_r12=0
        ep_r13=0
        ep_r14=0
        ep_r15=0
        
        buffer_r0=[]
        buffer_r1=[]
        buffer_r2=[]
        buffer_r3=[]
        buffer_r4=[]
        buffer_r5=[]
        buffer_r6=[]
        buffer_r7=[]
        buffer_r8=[]
        buffer_r9=[]
        buffer_r10=[]
        buffer_r11=[]
        buffer_r12=[]
        buffer_r13=[]
        buffer_r14=[]
        buffer_r15=[]
        
        
        chusudu=20
#        chusudu=25
        print(chusudu)
        for i in range(0,40):
            leading.append(0)
        for i in range(40,70):
            leading.append(-1)
        for i in range(70,150):
            leading.append(1)
#        for i in range(150,300):
#            leading.append(1)
            
        for step in range(100):
            exist_list=traci.vehicle.getIDList()
            if len(exist_list)>0:
                traci.vehicle.setSpeed(exist_list[0],chusudu)
            traci.simulationStep()
        
        for i in range(16):
            gapvector[i]=2*chusudu
#        gapvector=[2*chusudu]*16
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
#        print(leading)
        kexingjiasudu=[3]*Platoonsize
        for t in range(150):    # in one episode仿真运行一次
#            print(t)
#            print(initialpo)
            
            exist_list=traci.vehicle.getIDList()
            initialsp=[]
            if t==0:
                po=[]
                sp=[]
                acc=[]
            for xx in exist_list:
                traci.vehicle.setSpeedMode(xx,speedmode)
                initialsp.append(traci.vehicle.getSpeed(xx))
                if t==0:
                    po.append(traci.vehicle.getPosition(xx)[0])
                    sp.append(traci.vehicle.getSpeed(xx))
                    acc.append(traci.vehicle.getAcceleration(xx))
            
            acc_matrix=[]
#            print(t,po)
            
            
#            cuole=0
#            print(po)
            for i in range(Platoonsize):
#                if po[i]>5000 and po[i]<6000 and i<3:
                locationplot.append(po[i]/1000)
                speedplot.append(sp[i])
                timeplot.append(t)
#                if t==0:
#                    print(po[i]-po[i+1]-5)
#                print(t,i,(po[i]-po[i+1]-5)/(sp[i+1]+0.01))
#            if len(po)<Platoonsize+1:
#                print('baocuo1111111111111111111111111111111111')
#                cuole=1
#                traci.close()
#                sys.stdout.flush()
#                break

#            print(sp)
            for i in range(Platoonsize):
                if sp[i]-3<sp[i+1]:
                    gap=po[i]-po[i+1]-5-sp[i+1]+max(sp[i]-3,0)                        
                    if gap<0:
                        amax=-3
#                        print(gap)
                    else:
#                        amax=math.sqrt(madr*gap)+sp[i]-sp[i+1]-3
                        amax=min(gap/3,math.sqrt(madr*gap))+sp[i]-sp[i+1]-3
                        amax=np.clip(amax,-3,3)
                else:
                    amax=3
                kexingjiasudu[i]=amax
#            print(t,round(kexingjiasudu[0],1),round(kexingjiasudu[1],1))
            s0.append(calcstate(0,po,sp,acc,kexingjiasudu[0]))
            s1.append(calcstate(1,po,sp,acc,kexingjiasudu[1]))
            s2.append(calcstate(2,po,sp,acc,kexingjiasudu[2]))
            s3.append(calcstate(3,po,sp,acc,kexingjiasudu[3]))
            s4.append(calcstate(4,po,sp,acc,kexingjiasudu[4]))
            s5.append(calcstate(5,po,sp,acc,kexingjiasudu[5]))
            s6.append(calcstate(6,po,sp,acc,kexingjiasudu[6]))
            s7.append(calcstate(7,po,sp,acc,kexingjiasudu[7]))
            s8.append(calcstate(8,po,sp,acc,kexingjiasudu[8]))
            s9.append(calcstate(9,po,sp,acc,kexingjiasudu[9]))
            s10.append(calcstate(10,po,sp,acc,kexingjiasudu[10]))
            s11.append(calcstate(11,po,sp,acc,kexingjiasudu[11]))
            s12.append(calcstate(12,po,sp,acc,kexingjiasudu[12]))
            s13.append(calcstate(13,po,sp,acc,kexingjiasudu[13]))
            s14.append(calcstate(14,po,sp,acc,kexingjiasudu[14]))
            s15.append(calcstate(15,po,sp,acc,kexingjiasudu[15]))
#            if ep>100 and t<400:
            

            a0.append(ppo.choose_action(np.array(s0[-1])))#np转换之后就没有逗号了
            a1.append(ppo.choose_action(np.array(s1[-1])))
            a2.append(ppo.choose_action(np.array(s2[-1])))
            a3.append(ppo.choose_action(np.array(s3[-1])))#np转换之后就没有逗号了
            a4.append(ppo.choose_action(np.array(s4[-1])))
            a5.append(ppo.choose_action(np.array(s5[-1])))
            a6.append(ppo.choose_action(np.array(s6[-1])))
            a7.append(ppo.choose_action(np.array(s7[-1])))
            a8.append(ppo.choose_action(np.array(s8[-1])))#np转换之后就没有逗号了
            a9.append(ppo.choose_action(np.array(s9[-1])))
            a10.append(ppo.choose_action(np.array(s10[-1])))
            a11.append(ppo.choose_action(np.array(s11[-1])))#np转换之后就没有逗号了
            a12.append(ppo.choose_action(np.array(s12[-1])))
            a13.append(ppo.choose_action(np.array(s13[-1])))
            a14.append(ppo.choose_action(np.array(s14[-1])))
            a15.append(ppo.choose_action(np.array(s15[-1])))

            acc_matrix.append(a0[-1][0])
            acc_matrix.append(a1[-1][0])
            acc_matrix.append(a2[-1][0])
            acc_matrix.append(a3[-1][0])
            acc_matrix.append(a4[-1][0])
            acc_matrix.append(a5[-1][0])
            acc_matrix.append(a6[-1][0])
            acc_matrix.append(a7[-1][0])
            acc_matrix.append(a8[-1][0])
            acc_matrix.append(a9[-1][0])
            acc_matrix.append(a10[-1][0])
            acc_matrix.append(a11[-1][0])
            acc_matrix.append(a12[-1][0])
            acc_matrix.append(a13[-1][0])
            acc_matrix.append(a14[-1][0])
            acc_matrix.append(a15[-1][0])
#            print(t,acc_matrix)

            if t>250 and t<255:
                print(round(a0[-1][0],2)*1.5,round(a1[-1][0],2)*1.5)
            xiayimiaosuduvector=[0]*(Platoonsize+1)
            
            xiayimiaosudu=np.clip(sp[0]+touche[t],0,chusudu)
            traci.vehicle.setSpeed(exist_list[0],xiayimiaosudu)
            chaoguo=[0]*Platoonsize
            chongtu=[0]*Platoonsize
            for i in range(Platoonsize):
                accc=min(acc_matrix[i]*1.5,kexingjiasudu[i])
                if acc_matrix[i]*1.5>kexingjiasudu[i]+0.5:
                    chaoguo[i]=1
                xiayimiaosudu=np.clip(sp[i+1]+accc,0,33)
                if i<2**(1+ep//xunlian):
                    traci.vehicle.setSpeed(exist_list[i+1],xiayimiaosudu)
            traci.simulationStep()
              
            po=[]
            sp=[]
            acc=[]
            endsp=[]
            for xx in exist_list:
                po.append(traci.vehicle.getPosition(xx)[0])
                sp.append(round(traci.vehicle.getSpeed(xx),1))
                acc.append(traci.vehicle.getAcceleration(xx))
                endsp.append(traci.vehicle.getSpeed(xx))
                
            for i in range(Platoonsize):
                if sp[i]-3<sp[i+1]:
                    gap=po[i]-po[i+1]-5-sp[i+1]+max(sp[i]-3,0)                        
                    if gap<0:
                        amax=-3
#                        print(gap)
                    else:
#                        amax=math.sqrt(madr*gap)+sp[i]-sp[i+1]-3
                        amax=min(gap/3,math.sqrt(madr*gap))+sp[i]-sp[i+1]-3
                        amax=np.clip(amax,-3,3)
                else:
                    amax=3
                kexingjiasudu[i]=amax

            for i in range(len(sp)):
                if i>0 and (po[i]>po[i-1]-5 or po[i]<-10000):
                    chongtu[i-1]=1 

            for i in range(1,Platoonsize+1):        
                v=sp[i]
                a=acc[i]
                if a+0.5*Cd*rouair*Af*v*v/m+miu*g>=0:
                    fv=0.1569+0.0245*v-0.0007415*v*v+0.00005975*v**3+a*(0.07224+0.09681*v+0.001075*v*v)
                else:
                    fv=0
                consumption+=fv
            
            s_0.append(calcstate(0,po,sp,acc,kexingjiasudu[0]))
            r0.append(calcreward(0,po,sp,acc,chongtu[0],chaoguo[0]))
            buffer_r0.append((r0[-1]+ave)/ave)
            ep_r0 += r0[-1]
            
            s_1.append(calcstate(1,po,sp,acc,kexingjiasudu[1]))
            r1.append(calcreward(1,po,sp,acc,chongtu[1],chaoguo[1]))
            buffer_r1.append((r1[-1]+ave)/ave)
            ep_r1 += r1[-1]
            
            s_2.append(calcstate(2,po,sp,acc,kexingjiasudu[2]))
            r2.append(calcreward(2,po,sp,acc,chongtu[2],chaoguo[2]))
            buffer_r2.append((r2[-1]+ave)/ave)
            ep_r2 += r2[-1]
            
            s_3.append(calcstate(3,po,sp,acc,kexingjiasudu[3]))
            r3.append(calcreward(3,po,sp,acc,chongtu[3],chaoguo[3]))
            buffer_r3.append((r3[-1]+ave)/ave)
            ep_r3 += r3[-1]
            
            
            s_4.append(calcstate(4,po,sp,acc,kexingjiasudu[4]))
            r4.append(calcreward(4,po,sp,acc,chongtu[4],chaoguo[4]))
            buffer_r4.append((r4[-1]+ave)/ave)
            ep_r4 += r4[-1]
            
            s_5.append(calcstate(5,po,sp,acc,kexingjiasudu[5]))
            r5.append(calcreward(5,po,sp,acc,chongtu[5],chaoguo[5]))
            buffer_r5.append((r5[-1]+ave)/ave)
            ep_r5 += r5[-1]

            s_6.append(calcstate(6,po,sp,acc,kexingjiasudu[6]))
            r6.append(calcreward(6,po,sp,acc,chongtu[6],chaoguo[6]))
            buffer_r6.append((r6[-1]+ave)/ave)
            ep_r6 += r6[-1]
            
            s_7.append(calcstate(7,po,sp,acc,kexingjiasudu[7]))
            r7.append(calcreward(7,po,sp,acc,chongtu[7],chaoguo[7]))
            buffer_r7.append((r7[-1]+ave)/ave)
            ep_r7 += r7[-1]
            
            s_8.append(calcstate(8,po,sp,acc,kexingjiasudu[8]))
            r8.append(calcreward(8,po,sp,acc,chongtu[8],chaoguo[8]))
            buffer_r8.append((r8[-1]+ave)/ave)
            ep_r8 += r8[-1]
            
            s_9.append(calcstate(9,po,sp,acc,kexingjiasudu[9]))
            r9.append(calcreward(9,po,sp,acc,chongtu[9],chaoguo[9]))
            buffer_r9.append((r9[-1]+ave)/ave)
            ep_r9 += r9[-1]
            
            s_10.append(calcstate(10,po,sp,acc,kexingjiasudu[10]))
            r10.append(calcreward(10,po,sp,acc,chongtu[10],chaoguo[10]))
            buffer_r10.append((r10[-1]+ave)/ave)
            ep_r10 += r10[-1]
            
            s_11.append(calcstate(11,po,sp,acc,kexingjiasudu[11]))
            r11.append(calcreward(11,po,sp,acc,chongtu[11],chaoguo[11]))
            buffer_r11.append((r11[-1]+ave)/ave)
            ep_r11 += r11[-1]
            
            s_12.append(calcstate(12,po,sp,acc,kexingjiasudu[12]))
            r12.append(calcreward(12,po,sp,acc,chongtu[12],chaoguo[12]))
            buffer_r12.append((r12[-1]+ave)/ave)
            ep_r12 += r12[-1]
            
            s_13.append(calcstate(13,po,sp,acc,kexingjiasudu[13]))
            r13.append(calcreward(13,po,sp,acc,chongtu[13],chaoguo[13]))
            buffer_r13.append((r13[-1]+ave)/ave)
            ep_r13 += r13[-1]

            s_14.append(calcstate(14,po,sp,acc,kexingjiasudu[14]))
            r14.append(calcreward(14,po,sp,acc,chongtu[14],chaoguo[14]))
            buffer_r14.append((r14[-1]+ave)/ave)
            ep_r14 += r14[-1]
            
            s_15.append(calcstate(15,po,sp,acc,kexingjiasudu[15]))
            r15.append(calcreward(15,po,sp,acc,chongtu[15],chaoguo[15]))
            buffer_r15.append((r15[-1]+ave)/ave)
            ep_r15 += r15[-1]
            
            

            if ((t+1) % BATCH == 0 or t == EP_LEN-1) or sum(chongtu)>0:

                v_s_0 = ppo.get_v(np.array(s_0[-1]))
                discounted_r0= []
#                print(buffer_r0[::-1])
                for r in buffer_r0[::-1]:
                    v_s_0 = r + GAMMA * v_s_0
                    discounted_r0.append(v_s_0)
                discounted_r0.reverse()
                
                v_s_1 = ppo.get_v(np.array(s_1[-1]))
                discounted_r1= []
#                print(buffer_r1[::-1])
                for r in buffer_r1[::-1]:
                    v_s_1 = r + GAMMA * v_s_1
                    discounted_r1.append(v_s_1)
                discounted_r1.reverse()
                
                v_s_2 = ppo.get_v(np.array(s_2[-1]))
                discounted_r2= []
                for r in buffer_r2[::-1]:
                    v_s_2 = r + GAMMA * v_s_2
                    discounted_r2.append(v_s_2)
                discounted_r2.reverse()
                
                v_s_3 = ppo.get_v(np.array(s_3[-1]))
                discounted_r3= []
                for r in buffer_r3[::-1]:
                    v_s_3 = r + GAMMA * v_s_3
                    discounted_r3.append(v_s_3)
                discounted_r3.reverse()
                
                v_s_4 = ppo.get_v(np.array(s_4[-1]))
                discounted_r4= []
                for r in buffer_r4[::-1]:
                    v_s_4 = r + GAMMA * v_s_4
                    discounted_r4.append(v_s_4)
                discounted_r4.reverse()
                
                
                v_s_5 = ppo.get_v(np.array(s_5[-1]))
                discounted_r5= []
                for r in buffer_r5[::-1]:
                    v_s_5 = r + GAMMA * v_s_5
                    discounted_r5.append(v_s_5)
                discounted_r5.reverse()
                
                v_s_6 = ppo.get_v(np.array(s_6[-1]))
                discounted_r6= []
                for r in buffer_r6[::-1]:
                    v_s_6 = r + GAMMA * v_s_6
                    discounted_r6.append(v_s_6)
                discounted_r6.reverse()
                
                v_s_7 = ppo.get_v(np.array(s_7[-1]))
                discounted_r7= []
                for r in buffer_r7[::-1]:
                    v_s_7 = r + GAMMA * v_s_7
                    discounted_r7.append(v_s_7)
                discounted_r7.reverse()
                
                v_s_8 = ppo.get_v(np.array(s_8[-1]))
                discounted_r8= []
                for r in buffer_r8[::-1]:
                    v_s_8 = r + GAMMA * v_s_8
                    discounted_r8.append(v_s_8)
                discounted_r8.reverse()
                
                v_s_9 = ppo.get_v(np.array(s_9[-1]))
                discounted_r9= []
                for r in buffer_r9[::-1]:
                    v_s_9= r + GAMMA * v_s_9
                    discounted_r9.append(v_s_9)
                discounted_r9.reverse()
                
                v_s_10 = ppo.get_v(np.array(s_10[-1]))
                discounted_r10= []
                for r in buffer_r10[::-1]:
                    v_s_10 = r + GAMMA * v_s_10
                    discounted_r10.append(v_s_10)
                discounted_r10.reverse()
                
                v_s_11 = ppo.get_v(np.array(s_11[-1]))
                discounted_r11= []
                for r in buffer_r11[::-1]:
                    v_s_11 = r + GAMMA * v_s_11
                    discounted_r11.append(v_s_11)
                discounted_r11.reverse()
                
                v_s_12 = ppo.get_v(np.array(s_12[-1]))
                discounted_r12= []
                for r in buffer_r12[::-1]:
                    v_s_12 = r + GAMMA * v_s_12
                    discounted_r12.append(v_s_12)
                discounted_r12.reverse()
                
                
                v_s_13 = ppo.get_v(np.array(s_13[-1]))
                discounted_r13= []
                for r in buffer_r13[::-1]:
                    v_s_13 = r + GAMMA * v_s_13
                    discounted_r13.append(v_s_13)
                discounted_r13.reverse()
                
                v_s_14 = ppo.get_v(np.array(s_14[-1]))
                discounted_r14= []
                for r in buffer_r14[::-1]:
                    v_s_14 = r + GAMMA * v_s_14
                    discounted_r14.append(v_s_14)
                discounted_r14.reverse()
                
                v_s_15 = ppo.get_v(np.array(s_15[-1]))
                discounted_r15= []
                for r in buffer_r15[::-1]:
                    v_s_15 = r + GAMMA * v_s_15
                    discounted_r15.append(v_s_15)
                discounted_r15.reverse()
                
#                print(s0,a0,discounted_r0)
                buffer_s=s0+s1
                buffer_a=a0+a1
                discounted_r=discounted_r0+discounted_r1
                
                if ep//xunlian>0:
                    buffer_s=buffer_s+s2+s3
                    buffer_a=buffer_a+a2+a3
                    discounted_r=discounted_r+discounted_r2+discounted_r3
                if ep//xunlian>1:
                    buffer_s=buffer_s+s4+s5+s6+s7
                    buffer_a=buffer_a+a4+a5+a6+a7
                    discounted_r=discounted_r+discounted_r4+discounted_r5+discounted_r6+discounted_r7
                if ep//xunlian>2:
                    buffer_s=buffer_s+s8+s9+s10+s11+s12+s13+s14+s15
                    buffer_a=buffer_a+a8+a9+a10+a11+a12+a13+a14+a15
                    discounted_r=discounted_r+discounted_r8+discounted_r9+discounted_r10+discounted_r11+discounted_r12+discounted_r13+discounted_r14+discounted_r15
                    
  
                bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]  
#                print(t,ba,br)
                buffer_s, buffer_a, buffer_r = [], [], []#到buffer这里再堆起来
                ppo.update(bs, ba, br)
            if sum(chongtu)>0:
                traci.close()
                sys.stdout.flush()
                print('baocuo222222222222222222222222222222222')
                print(t,chongtu)
                break
#        print(cuole)
        if sum(chongtu)>0:
            continue
        zuihouweizhi=[]
        for xx in exist_list:
            zuihouweizhi.append(traci.vehicle.getPosition(xx)[0])
        distance=0 
        for i in range(Platoonsize+1):
            distance+=(zuihouweizhi[i]-chushiweizhi[i])
        distance+=1

#        print('fuel',consumption/distance*100)
        
        juli.append(distance)
        epr=(ep_r0+ep_r1+ep_r2+ep_r3+ep_r4+ep_r5+ep_r6+ep_r7+ep_r8+ep_r9+ep_r10+ep_r11+ep_r12+ep_r13+ep_r14+ep_r15)/160000*2
        print(ep,epr)




        if len(rewardvector) == 0: 
            rewardvector.append(epr)
        else: 
            rewardvector.append(rewardvector[-1]*0.8 + epr*0.2)
#        rewardvector.append(epr)
        plt.ion()
        plt.figure(ep*2-2)
        plt.plot(np.arange(len(rewardvector)), rewardvector)
        plt.xlabel('Episode');plt.ylabel('Episode reward')
        plt.draw()
        plt.pause(1)
        plt.close()#越大越好
        
        
        traci.close()
        sys.stdout.flush()
        
        plt.ion()
        plt.figure(ep*2-1)
        plt.scatter(timeplot,locationplot,c=speedplot,s=10,alpha=0.3)
        plt.colorbar()
        plt.xlabel('Time (s)');plt.ylabel('Location (km)')
        plt.grid(True)
        plt.show()
        
        if ep<50:
            print(rewardvector[-1])
        
        
    
    save_path=ppo.saver.save(ppo.sess,"model/model.ckpt")
    print("save to path:",save_path)
    