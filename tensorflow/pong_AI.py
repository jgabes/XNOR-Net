import numpy as np
import config
import nets
import gym
import tensorflow as tf
from matplotlib import pyplot as plt
import time

plot = True
load_start = True
def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float32)


def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    running_add = running_add * config.gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r


with tf.Session() as sess:
    in_img = tf.placeholder( tf.float32,[None, config.tf_shape[1], config.tf_shape[2], config.tf_shape[3]])
    lab = tf.placeholder( tf.float32,[None, 1])
    discount = tf.placeholder(tf.float32, [None,1])
    out = nets.pong_norm(in_img)
    err = tf.reduce_sum(tf.losses.absolute_difference(labels = lab, predictions = out, weights=discount))



    optimizer = tf.train.RMSPropOptimizer(config.lr)
    train_op = optimizer.minimize(err)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    saver = tf.train.Saver()
    last_mean_perf = -21

    if load_start:
        saver.restore(sess, "./normal_RL.ckpt")
        last_mean_perf=float(np.loadtxt("./old_mean_perf.txt"))
        print("loaded checkpoint" )


    env = gym.make("Pong-v0")
    observation = env.reset()
    prev_x = None # used in computing the difference frame
    running_reward = None
    reward_sum = 0
    episode_number = 0

    x_buffer = []
    out_buffer = []
    reward_buffer = []
    y_buffer = []
    action_buffer = []
    slow = True
    all_rewards = []
    render = True

    while True:

        if render: env.render()

        # preprocess the observation, set input to network to be difference image
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(config.pong_shape)
        #plt.imshow(x)
        #plt.pause(0.001)
        x_buffer.append(x)
        prev_x = cur_x

        aprob = sess.run(out, feed_dict={in_img: np.reshape(x, config.tf_shape)}).flatten()

        out_buffer.append(aprob)
        action = 2 if np.random.uniform() < aprob else 3  # roll the dice!
        action_buffer.append(action)
        y = 1 if action == 2 else 0
        y_buffer.append(y)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)

        observation, reward, done, info = env.step(action)
        reward_sum += reward

        reward_buffer.append(reward)
        if(slow):
            time.sleep(.002)

        if done:
            render=False
            slow = False
            plot=False
            #print(reward)# an episode finished
            episode_number += 1
            all_rewards.append(reward_sum)

            epx = np.vstack(x_buffer)
            epr = np.vstack(reward_buffer)
            epy = np.vstack(y_buffer)
            epo = np.vstack(out_buffer)
            discounted_epr = discount_rewards(epr)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
            discount_reward_label = epy * discounted_epr
            if plot:
                '''
                plt.figure(2)
                plt.clf()

                plt.plot(epr)
                plt.title("reward buffer")

                plt.figure(3)
                plt.clf()

                plt.plot(discounted_epr)
                plt.title("discounted reward")

                plt.figure(4)
                plt.clf()
                plt.plot(epo)
                plt.title("network output")

                plt.figure(5)
                plt.clf()
                plt.plot(epy)
                plt.title("action taken")

                plt.figure(6)
                plt.clf()

                plt.plot(epy-epo)
                plt.title("pre-discount-error signal")
'''
                plt.figure(7)
                plt.clf()
                plt.plot(epo)
                plt.plot(epy-epo)
                plt.plot(discount_reward_label)
                plt.pause(0.01)

            _ =sess.run(train_op, feed_dict={in_img: np.asarray(x_buffer).reshape([len(x_buffer), config.tf_shape[1], config.tf_shape[2], config.tf_shape[3]]),
                                             lab: epy,
                                             discount : discounted_epr} )
            x_buffer, out_buffer, reward_buffer, y_buffer, action_buffer = [], [], [], [], []
            if episode_number%100 ==0:
                render=True
                plot=True
                slow=True
                mean_perf = np.mean(np.asarray(all_rewards))
                if mean_perf>=last_mean_perf: #There has been some improvement
                    save_path = saver.save(sess, "./normal_RL.ckpt")
                    print("Model saved in path: %s" % save_path)
                    print("new mean perf:", mean_perf)
                    np.savetxt("./old_mean_perf.txt", np.asarray(mean_perf, np.int).reshape([1, 1]))
                    last_mean_perf= mean_perf
                else:
                    saver.restore(sess, "./normal_RL.ckpt")
                    print("learning went backwards, restored old checkpoint")
                all_rewards = []
            pass

            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
            reward_sum = 0
            observation = env.reset()  # reset env
            prev_x = None




