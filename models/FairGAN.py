"""
Code for the FairGAN model was provided by the DECAF authors as part of our
email correspondence with them. Original source code for the model seems to
be in the Medgan repository: https://github.com/mp2893/medgan

We made modifications to the original source code in order to make it compatible
with Tensorflow v2.
"""

import argparse
import sys
import time

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tf_slim import batch_norm, l2_regularizer

_VALIDATION_RATIO = 0.1
p_s = [0.32495245676882933, 0.6750475432311707]
lambda_fair = 1


class Medgan(object):
    def __init__(self,
                 dataType='binary',
                 inputDim=58,
                 embeddingDim=32,
                 randomDim=32,
                 generatorDims=(32, 32),
                 discriminatorDims=(32, 16, 1),
                 compressDims=(),
                 decompressDims=(),
                 bnDecay=0.99,
                 l2scale=0.001):
        self.inputDim = inputDim
        self.embeddingDim = embeddingDim
        self.generatorDims = list(generatorDims) + [embeddingDim]
        self.randomDim = randomDim
        self.dataType = dataType

        if dataType == 'binary':
            self.aeActivation = tf.compat.v1.nn.tanh
        else:
            self.aeActivation = tf.compat.v1.nn.relu

        self.generatorActivation = tf.compat.v1.nn.relu
        self.discriminatorActivation = tf.compat.v1.nn.relu
        self.discriminatorDims = discriminatorDims
        self.compressDims = list(compressDims) + [embeddingDim]
        self.decompressDims = list(decompressDims) + [inputDim]
        self.bnDecay = bnDecay
        self.l2scale = l2scale

    def loadData(self, dataPath=''):
        print(dataPath)
        data = np.load(dataPath, allow_pickle=True)

        if self.dataType == 'binary':
            data = np.clip(data, 0, 1)

        trainD, validD = train_test_split(data, test_size=_VALIDATION_RATIO, random_state=0)
        trainX = trainD[:,1:]
        trainz = trainD[:,0]
        validX = validD[:, 1:]
        validz = validD[:, 0]

        return trainX, validX, trainz, validz

    def buildAutoencoder(self, x_input):
        decodeVariables = {}
        with tf.compat.v1.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2scale)):
            tempVec = x_input
            tempDim = self.inputDim
            i = 0
            for compressDim in self.compressDims:
                W = tf.compat.v1.get_variable('aee_W_'+str(i), shape=[tempDim, compressDim])
                b = tf.compat.v1.get_variable('aee_b_'+str(i), shape=[compressDim])
                tempVec = self.aeActivation(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, W), b))
                tempDim = compressDim
                i += 1
    
            i = 0
            for decompressDim in self.decompressDims[:-1]:
                W = tf.compat.v1.get_variable('aed_W_'+str(i), shape=[tempDim, decompressDim])
                b = tf.compat.v1.get_variable('aed_b_'+str(i), shape=[decompressDim])
                tempVec = self.aeActivation(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, W), b))
                tempDim = decompressDim
                decodeVariables['aed_W_'+str(i)] = W
                decodeVariables['aed_b_'+str(i)] = b
                i += 1
            W = tf.compat.v1.get_variable('aed_W_'+str(i), shape=[tempDim, self.decompressDims[-1]])
            b = tf.compat.v1.get_variable('aed_b_'+str(i), shape=[self.decompressDims[-1]])
            decodeVariables['aed_W_'+str(i)] = W
            decodeVariables['aed_b_'+str(i)] = b

            if self.dataType == 'binary':
                x_reconst = tf.compat.v1.nn.sigmoid(tf.compat.v1.add(tf.compat.v1.matmul(tempVec,W),b))
                loss = tf.compat.v1.reduce_mean(-tf.compat.v1.reduce_sum(x_input * tf.compat.v1.log(x_reconst + 1e-12) + (1. - x_input) * tf.compat.v1.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.compat.v1.nn.relu(tf.compat.v1.add(tf.compat.v1.matmul(tempVec,W),b))
                loss = tf.compat.v1.reduce_mean((x_input - x_reconst)**2)
            
        return loss, decodeVariables

    def buildGenerator(self, x_input, z_input, bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        embeddings = tf.compat.v1.Variable(tf.compat.v1.random_uniform([2, tempDim], -1.0, 1.0))
        embed = tf.compat.v1.nn.embedding_lookup(embeddings, z_input)
        tempVec = tf.compat.v1.multiply(tempVec, embed)

        with tf.compat.v1.variable_scope('generator', regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.compat.v1.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.compat.v1.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.compat.v1.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.compat.v1.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None)

            if self.dataType == 'binary':
                h3 = tf.compat.v1.nn.tanh(h2)
            else:
                h3 = tf.compat.v1.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def buildGeneratorTest(self, x_input, z_input,  bn_train):
        tempVec = x_input
        tempDim = self.randomDim
        embeddings = tf.compat.v1.Variable(tf.compat.v1.random_uniform([2, tempDim], -1.0, 1.0))
        embed = tf.compat.v1.nn.embedding_lookup(embeddings, z_input)
        tempVec = tf.compat.v1.multiply(tempVec, embed)

        with tf.compat.v1.variable_scope('generator', regularizer=l2_regularizer(self.l2scale)):
            for i, genDim in enumerate(self.generatorDims[:-1]):
                W = tf.compat.v1.get_variable('W_'+str(i), shape=[tempDim, genDim])
                h = tf.compat.v1.matmul(tempVec,W)
                h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)
                h3 = self.generatorActivation(h2)
                tempVec = h3 + tempVec
                tempDim = genDim
            W = tf.compat.v1.get_variable('W'+str(i), shape=[tempDim, self.generatorDims[-1]])
            h = tf.compat.v1.matmul(tempVec,W)
            h2 = batch_norm(h, decay=self.bnDecay, scale=True, is_training=bn_train, updates_collections=None, trainable=False)

            if self.dataType == 'binary':
                h3 = tf.compat.v1.nn.tanh(h2)
            else:
                h3 = tf.compat.v1.nn.relu(h2)

            output = h3 + tempVec
        return output
    
    def getDiscriminatorResults(self, x_input, z_bool, keepRate, reuse=False):
        batchSize = tf.compat.v1.shape(x_input)[0]
        inputMean = tf.compat.v1.reshape(tf.compat.v1.tile(tf.compat.v1.reduce_mean(x_input,0), [batchSize]), (batchSize, self.inputDim))
        tempVec = tf.compat.v1.concat([x_input, inputMean], 1)
        tempDim = self.inputDim * 2
        with tf.compat.v1.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            for i, discDim in enumerate(self.discriminatorDims[:-1]):
                W = tf.compat.v1.get_variable('W_'+str(i), shape=[tempDim, discDim])
                b = tf.compat.v1.get_variable('b_'+str(i), shape=[discDim])
                h = self.discriminatorActivation(tf.compat.v1.add(tf.compat.v1.matmul(tempVec,W),b))
                h = tf.compat.v1.nn.dropout(h, keepRate)
                tempVec = h
                tempDim = discDim
            W0 = tf.compat.v1.get_variable('W0', shape=[tempDim, 1])
            b0 = tf.compat.v1.get_variable('b0', shape=[1])
            W1 = tf.compat.v1.get_variable('W1', shape=[tempDim, 1])
            b1 = tf.compat.v1.get_variable('b1', shape=[1])
            y_hat = tf.compat.v1.where (z_bool,tf.compat.v1.squeeze(tf.compat.v1.nn.sigmoid(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, W1), b1))),tf.compat.v1.squeeze(tf.compat.v1.nn.sigmoid(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, W0), b0))))
            W_z = tf.compat.v1.get_variable('W_z', shape=[tempDim, 1])
            b_z = tf.compat.v1.get_variable('b_z', shape=[1])
            z_hat = tf.compat.v1.squeeze(tf.compat.v1.nn.sigmoid(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, W_z), b_z)))
        return y_hat, z_hat
    
    def buildDiscriminator(self, x_real, z_real, x_fake, z_fake, zb_real, zb_fake, keepRate, decodeVariables, bn_train):
        #Discriminate for real samples
        y_hat_real, z_hat_real = self.getDiscriminatorResults(x_real, zb_real, keepRate, reuse=False)

        #Decompress, then discriminate for real samples
        tempVec = x_fake
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_decoded = tf.compat.v1.nn.sigmoid(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_decoded = tf.compat.v1.nn.relu(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        y_hat_fake, z_hat_fake = self.getDiscriminatorResults(x_decoded, zb_fake, keepRate, reuse=True)

        loss_d = -tf.compat.v1.reduce_mean(tf.compat.v1.log(y_hat_real + 1e-12)) - tf.compat.v1.reduce_mean(tf.compat.v1.log(1. - y_hat_fake + 1e-12))
        loss_g = -tf.compat.v1.reduce_mean(tf.compat.v1.log(y_hat_fake + 1e-12))
        loss_d_z = -tf.compat.v1.reduce_mean(z_real * tf.compat.v1.log(z_hat_real + 1e-12) + z_fake * tf.compat.v1.log(z_hat_fake + 1e-12)) - tf.compat.v1.reduce_mean((1. - z_real) * tf.compat.v1.log(1. - z_hat_real + 1e-12) + (1. - z_fake) * tf.compat.v1.log( 1. - z_hat_fake + 1e-12))
        loss_g_z = -tf.compat.v1.reduce_mean((1. - z_fake) * tf.compat.v1.log(z_hat_fake + 1e-12)) - tf.compat.v1.reduce_mean(z_fake * tf.compat.v1.log( 1. - z_hat_fake + 1e-12))
        loss_d_total = -tf.compat.v1.reduce_mean(tf.compat.v1.log(y_hat_real + 1e-12) + tf.compat.v1.log(1. - y_hat_fake + 1e-12) +
                                       lambda_fair* ( z_real * tf.compat.v1.log(z_hat_real + 1e-12) + z_fake * tf.compat.v1.log(z_hat_fake + 1e-12) +
            (1. - z_real) * tf.compat.v1.log(1. - z_hat_real + 1e-12) + (1. - z_fake) * tf.compat.v1.log(1. - z_hat_fake + 1e-12)))
        loss_g_total = -tf.compat.v1.reduce_mean( lambda_fair*( (1. - z_fake) * tf.compat.v1.log(z_hat_fake + 1e-12) + z_fake * tf.compat.v1.log(1. - z_hat_fake + 1e-12) )  + tf.compat.v1.log(y_hat_fake + 1e-12) )

        return x_decoded, loss_d_total, loss_g_total, loss_d, loss_d_z, loss_g, loss_g_z, y_hat_real, z_hat_real, y_hat_fake, z_hat_fake

    def print2file(self, buf, outFile):
        outfd = open(outFile, 'a')
        outfd.write(buf + '\n')
        outfd.close()
    
    def generateData(self,
                     nSamples=100,
                     modelFile='model',
                     batchSize=100,
                     outFile='out'):
        x_dummy = tf.compat.v1.placeholder('float', [None, self.inputDim])
        _, decodeVariables = self.buildAutoencoder(x_dummy)
        x_random = tf.compat.v1.placeholder('float', [None, self.randomDim])
        z_random_idx = tf.compat.v1.placeholder('int32', [None])
        bn_train = tf.compat.v1.placeholder('bool')
        x_emb = self.buildGeneratorTest(x_random, z_random_idx, bn_train)
        tempVec = x_emb
        i = 0
        for _ in self.decompressDims[:-1]:
            tempVec = self.aeActivation(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
            i += 1

        if self.dataType == 'binary':
            x_reconst = tf.compat.v1.nn.sigmoid(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))
        else:
            x_reconst = tf.compat.v1.nn.relu(tf.compat.v1.add(tf.compat.v1.matmul(tempVec, decodeVariables['aed_W_'+str(i)]), decodeVariables['aed_b_'+str(i)]))

        np.random.seed(0)
        saver = tf.compat.v1.train.Saver()
        outputVec = []
        outputVec_z = []
        burn_in = 1000
        with tf.compat.v1.Session() as sess:
            saver.restore(sess, modelFile)
            print('burning in')
            for i in range(burn_in):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                randomz = np.random.choice([0,1], batchSize, p=p_s)
                output = sess.run(x_reconst, feed_dict={x_random:randomX, z_random_idx:randomz, bn_train:True})

            print('generating')
            nBatches = int(np.ceil(float(nSamples)) / float(batchSize))
            for i in range(nBatches):
                randomX = np.random.normal(size=(batchSize, self.randomDim))
                randomz = np.random.choice([0,1], batchSize, p=p_s)
                output = sess.run(x_reconst, feed_dict={x_random:randomX, z_random_idx:randomz, bn_train:False})
                outputVec.extend(output)
                outputVec_z.extend(randomz)

        outputMat = np.array(outputVec)
        outputMat_z = np.array(outputVec_z)
        np.save(outFile, outputMat)
        outFile_z = outFile + '_z'
        np.save(outFile_z, outputMat_z)
        return outputMat
    
    def calculateDiscAuc(self, preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc
    
    def calculateDiscAccuracy(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real: 
            if pred > 0.5: hit += 1
        for pred in preds_fake: 
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
        return acc

    def calculateGenAccuracy(self, preds_fake):
        total = len(preds_fake)
        hit = 0
        for pred in preds_fake:
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
        return acc

    def calculateDiscAuc_z(self, preds_real, preds_fake, z_real, z_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([z_real, z_fake], axis=0)
        auc = roc_auc_score(labels, preds)
        auc_real = roc_auc_score(z_real, preds_real)
        auc_fake = roc_auc_score(z_fake, preds_fake)
        return auc, auc_real, auc_fake

    def calculateDiscAccuracy_z(self, preds_real, preds_fake, z_real, z_fake):
        total = len(preds_real) + len(preds_fake)
        hit_real = 0
        hit_fake = 0
        for inx in range(len(preds_real)):
            if preds_real[inx] > 0.5 and z_real[inx] == 1: hit_real += 1
            if preds_real[inx] < 0.5 and z_real[inx] == 0: hit_real += 1
        for inx in range(len(preds_fake)):
            if preds_fake[inx] > 0.5 and z_fake[inx] == 1: hit_fake += 1
            if preds_fake[inx] < 0.5 and z_fake[inx] == 0: hit_fake += 1
        acc_real = float(hit_real) / float(len(preds_real))
        acc_fake = float(hit_fake) / float(len(preds_fake))
        acc = float(hit_real + hit_fake) / float(total)
        return acc, acc_real, acc_fake

    def calculateRD(self, gen_fake, z_real):
        income1 = gen_fake[:, -1].tolist().count(1)
        income0 = gen_fake.shape[0] - income1
        sex1 = z_real.tolist().count(1)
        sex0 = z_real.shape[0] - sex1
        sex_income = list(map(lambda x, y: (x, y), z_real.tolist(), gen_fake[:, -1].tolist()))
        si11 = sex_income.count((1, 1))
        si01 = sex_income.count((0, 1))
        rd = si01 / sex0 - si11 / sex1
        return rd

    def train(self,
              dataPath='data',
              modelPath='',
              outPath='out',
              nEpochs=500,
              discriminatorTrainPeriod=2,
              generatorTrainPeriod=1,
              pretrainBatchSize=100,
              batchSize=1000,
              pretrainEpochs=100,
              saveMaxKeep=0):
        x_raw = tf.compat.v1.placeholder('float', [None, self.inputDim])
        x_random= tf.compat.v1.placeholder('float', [None, self.randomDim])
        z_raw = tf.compat.v1.placeholder('float', [None])
        z_raw_bool = tf.compat.v1.placeholder('bool', [None])
        z_random = tf.compat.v1.placeholder('float', [None])
        z_random_idx = tf.compat.v1.placeholder('int32', [None])
        z_random_bool = tf.compat.v1.placeholder('bool', [None])
        keep_prob = tf.compat.v1.placeholder('float')
        bn_train = tf.compat.v1.placeholder('bool')

        loss_ae, decodeVariables = self.buildAutoencoder(x_raw)
        x_fake = self.buildGenerator(x_random, z_random_idx, bn_train)
        x_hat, loss_d_total, loss_g_total, loss_d, loss_d_z, loss_g, loss_g_z, y_hat_real, z_hat_real, y_hat_fake, z_hat_fake = self.buildDiscriminator(x_raw, z_raw, x_fake, z_random, z_raw_bool, z_random_bool, keep_prob, decodeVariables, bn_train)
        trainX, validX, trainz, validz = self.loadData(dataPath)

        t_vars = tf.compat.v1.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        all_regs = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

        optimize_ae = tf.compat.v1.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tf.compat.v1.train.AdamOptimizer().minimize(loss_d + sum(all_regs), var_list=d_vars)
        optimize_g = tf.compat.v1.train.AdamOptimizer().minimize(loss_g + sum(all_regs), var_list=g_vars+list(decodeVariables.values()))
        optimize_d_fair = tf.compat.v1.train.AdamOptimizer().minimize(loss_d_total + sum(all_regs), var_list=d_vars)
        optimize_g_fair = tf.compat.v1.train.AdamOptimizer().minimize(loss_g_total + sum(all_regs),
                                                       var_list=g_vars + list(decodeVariables.values()))

        initOp = tf.compat.v1.global_variables_initializer()

        nBatches = int(np.ceil(float(trainX.shape[0]) / float(batchSize)))
        saver = tf.compat.v1.train.Saver(max_to_keep=saveMaxKeep)
        logFile = outPath + '.log'

        with tf.compat.v1.Session() as sess:
            if modelPath == '': sess.run(initOp)
            else: saver.restore(sess, modelPath)
            nTrainBatches = int(np.ceil(float(trainX.shape[0])) / float(pretrainBatchSize))
            nValidBatches = int(np.ceil(float(validX.shape[0])) / float(pretrainBatchSize))

            if modelPath== '':
                for epoch in range(pretrainEpochs):
                    idx = np.random.permutation(trainX.shape[0])
                    trainLossVec = []
                    for i in range(nTrainBatches):
                        batchX = trainX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                        batchz = trainz[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                        _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw:batchX, z_raw:batchz})
                        trainLossVec.append(loss)
                    idx = np.random.permutation(validX.shape[0])
                    validLossVec = []
                    for i in range(nValidBatches):
                        batchX = validX[idx[i*pretrainBatchSize:(i+1)*pretrainBatchSize]]
                        batchz = validz[idx[i * pretrainBatchSize:(i + 1) * pretrainBatchSize]]
                        loss = sess.run(loss_ae, feed_dict={x_raw:batchX, z_raw:batchz})
                        validLossVec.append(loss)
                    validReverseLoss = 0.
                    buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f, validReverseLoss:%f' % (epoch, np.mean(trainLossVec), np.mean(validLossVec), validReverseLoss)
                    print(buf)
                    self.print2file(buf, logFile)


            for epoch in range(nEpochs):
                d_loss_vec= []
                g_loss_vec = []
                idx = np.arange(trainX.shape[0])
                for i in range(nBatches):
                    for _ in range(discriminatorTrainPeriod):
                        batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                        batchX = trainX[batchIdx]
                        batchz = trainz[batchIdx]
                        randomX = np.random.normal(size=(batchSize, self.randomDim))
                        randomz = np.random.choice([0,1], batchSize, p=p_s)
                        _, discLoss = sess.run([optimize_d, loss_d], feed_dict={x_raw:batchX, z_raw:batchz, z_raw_bool:batchz, x_random:randomX, z_random:randomz, z_random_idx:randomz, z_random_bool:randomz, keep_prob:1.0, bn_train:False})
                        d_loss_vec.append(discLoss)
                    for _ in range(generatorTrainPeriod):
                        randomX = np.random.normal(size=(batchSize, self.randomDim))
                        randomz = np.random.choice([0,1], batchSize, p=p_s)
                        _, generatorLoss = sess.run([optimize_g, loss_g], feed_dict={x_raw:batchX, z_raw:batchz, z_raw_bool:batchz, x_random:randomX, z_random:randomz, z_random_idx:randomz, z_random_bool:randomz,  keep_prob:1.0, bn_train:True})
                        g_loss_vec.append(generatorLoss)

                idx = np.arange(len(validX))
                nValidBatches = int(np.ceil(float(len(validX)) / float(batchSize)))
                validAccVec = []
                validAucVec = []
                validAccVec_g = []
                r_d_fake = []
                for i in range(nValidBatches):
                    batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                    batchX = validX[batchIdx]
                    batchz = validz[batchIdx]
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    randomz = np.random.choice([0,1], batchSize, p=p_s)
                    gen_fake, preds_real, preds_fake = sess.run([x_hat, y_hat_real, y_hat_fake], feed_dict={x_raw:batchX, z_raw:batchz, z_raw_bool:batchz, x_random:randomX, z_random:randomz, z_random_idx:randomz, z_random_bool:randomz, keep_prob:1.0, bn_train:False})
                    validAcc = self.calculateDiscAccuracy(preds_real, preds_fake)
                    validAcc_g = self.calculateGenAccuracy(preds_fake)
                    validAuc = self.calculateDiscAuc(preds_real, preds_fake)
                    rdf = self.calculateRD(gen_fake, randomz)
                    validAccVec.append(validAcc)
                    validAucVec.append(validAuc)
                    validAccVec_g.append(validAcc_g)
                    r_d_fake.append(rdf)
                buf = 'Epoch:%d, d_loss:%f, g_loss:%f, d accuracy:%f, d AUC:%f, g accuracy:%f, rdf %f' % (epoch, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(validAccVec), np.mean(validAucVec), np.mean(validAccVec_g), np.mean(r_d_fake))
                print(buf)
                self.print2file(buf, logFile+'_unfair')
            savePath = saver.save(sess, outPath+'_unfair')
            for epoch in range(nEpochs):
                d_loss_vec= []
                g_loss_vec = []
                d_loss_vec_z = []
                g_loss_vec_z = []
                d_loss_vec_total = []
                g_loss_vec_total = []
                idx = np.arange(trainX.shape[0])
                for i in range(nBatches):
                    for _ in range(discriminatorTrainPeriod):
                        batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                        batchX = trainX[batchIdx]
                        batchz = trainz[batchIdx]
                        randomX = np.random.normal(size=(batchSize, self.randomDim))
                        randomz = np.random.choice([0,1], batchSize, p=p_s)
                        _, disLoss_total, discLoss, disLoss_z = sess.run([optimize_d_fair, loss_d_total, loss_d, loss_d_z], feed_dict={x_raw:batchX, z_raw:batchz, z_raw_bool:batchz,  x_random:randomX, z_random:randomz, z_random_idx:randomz, z_random_bool:randomz, keep_prob:1.0, bn_train:False})
                        d_loss_vec.append(discLoss)
                        d_loss_vec_z.append(disLoss_z)
                        d_loss_vec_total.append(disLoss_total)
                    for _ in range(generatorTrainPeriod):
                        randomX = np.random.normal(size=(batchSize, self.randomDim))
                        randomz = np.random.choice([0,1], batchSize, p=p_s)
                        _, generatorLoss_total, generatorLoss, generatorLoss_z = sess.run([optimize_g_fair, loss_g_total, loss_g, loss_g_z], feed_dict={x_raw:batchX, z_raw:batchz, z_raw_bool:batchz, x_random:randomX, z_random:randomz, z_random_idx:randomz, z_random_bool:randomz, keep_prob:1.0, bn_train:True})
                        g_loss_vec.append(generatorLoss)
                        g_loss_vec_z.append(generatorLoss_z)
                        g_loss_vec_total.append(generatorLoss_total)

                idx = np.arange(len(validX))
                nValidBatches = int(np.ceil(float(len(validX)) / float(batchSize)))
                '''
                validAccVec = []
                validAucVec = []
                validAccVec_g = []
                r_d_fake = []
                validAccVec_z = []
                validAucVec_z = []
                validAccVec_z_real = []
                validAucVec_z_real = []
                validAccVec_z_fake = []
                validAucVec_z_fake = []
                for i in range(nValidBatches):
                    batchIdx = np.random.choice(idx, size=batchSize, replace=False)
                    batchX = validX[batchIdx]
                    batchz = validz[batchIdx]
                    randomX = np.random.normal(size=(batchSize, self.randomDim))
                    randomz = np.random.choice([0,1], batchSize, p=p_s)
                    gen_fake, preds_real, preds_real_z, preds_fake, preds_fake_z, = sess.run([x_hat, y_hat_real, z_hat_real, y_hat_fake, z_hat_fake], feed_dict={x_raw:batchX, z_raw:batchz, z_raw_bool:batchz, x_random:randomX, z_random:randomz, z_random_idx:randomz, z_random_bool:randomz, keep_prob:1.0, bn_train:False})
                    validAcc = self.calculateDiscAccuracy(preds_real, preds_fake)
                    validAcc_g = self.calculateGenAccuracy(preds_fake)
                    validAcc_z, validAcc_z_real, validAcc_z_fake = self.calculateDiscAccuracy_z(preds_real_z, preds_fake_z, batchz, randomz)
                    validAuc = self.calculateDiscAuc(preds_real, preds_fake)
                    rdf = self.calculateRD(gen_fake, randomz)
                    validAuc_z, validAuc_z_real, validAuc_z_fake = self.calculateDiscAuc_z(preds_real_z, preds_fake_z, batchz, randomz)
                    validAccVec.append(validAcc)
                    validAucVec.append(validAuc)
                    validAccVec_g.append(validAcc_g)
                    r_d_fake.append(rdf)
                    validAccVec_z.append(validAcc_z)
                    validAucVec_z.append(validAuc_z)
                    validAccVec_z_real.append(validAcc_z_real)
                    validAucVec_z_real.append(validAuc_z_real)
                    validAccVec_z_fake.append(validAcc_z_fake)
                    validAucVec_z_fake.append(validAuc_z_fake)
                buf = 'Epoch:%d, d_loss_total:%f g_loss_total:%f, d_loss:%f, g_loss:%f, d_accuracy:%f, d_AUC:%f, g_accuracy:%f, z_d_loss:%f, z_g_loss:%f, z_accuracy_total %f real %f fake %f, z_AUC_total %f real %f fake %f rdf %f' % (epoch, np.mean(d_loss_vec_total), np.mean(g_loss_vec_total), np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(validAccVec), np.mean(validAucVec), np.mean(validAccVec_g), np.mean(d_loss_vec_z), np.mean(g_loss_vec_z),  np.mean(validAccVec_z),  np.mean(validAccVec_z_real), np.mean(validAccVec_z_fake), np.mean(validAucVec_z), np.mean(validAucVec_z_real), np.mean(validAucVec_z_fake), np.mean(r_d_fake))
                print(buf)
                self.print2file(buf, logFile)
                '''

            savePath = saver.save(sess, outPath)
        print(savePath)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments(parser):
    parser.add_argument('--embed_size', type=int, default=128, help='The dimension size of the embedding, which will be generated by the generator. (default value: 128)')
    parser.add_argument('--noise_size', type=int, default=128, help='The dimension size of the random noise, on which the generator is conditioned. (default value: 128)')
    parser.add_argument('--generator_size', type=tuple, default=(128, 128), help='The dimension size of the generator. Note that another layer of size "--embed_size" is always added. (default value: (128, 128))')
    parser.add_argument('--discriminator_size', type=tuple, default=(256, 128, 1), help='The dimension size of the discriminator. (default value: (256, 128, 1))')
    parser.add_argument('--compressor_size', type=tuple, default=(), help='The dimension size of the encoder of the autoencoder. Note that another layer of size "--embed_size" is always added. Therefore this can be a blank tuple. (default value: ())')
    parser.add_argument('--decompressor_size', type=tuple, default=(), help='The dimension size of the decoder of the autoencoder. Note that another layer, whose size is equal to the dimension of the <patient_matrix>, is always added. Therefore this can be a blank tuple. (default value: ())')
    parser.add_argument('--data_type', type=str, default='binary', choices=['binary', 'count'], help='The input data type. The <patient matrix> could either contain binary values or count values. (default value: "binary")')
    parser.add_argument('--batchnorm_decay', type=float, default=0.99, help='Decay value for the moving average used in Batch Normalization. (default value: 0.99)')
    parser.add_argument('--L2', type=float, default=0.001, help='L2 regularization coefficient for all weights. (default value: 0.001)')

    parser.add_argument('--data_file', type=str, default='adult_large.pkl', help='The path to the numpy matrix containing aggregated patient records.')
    parser.add_argument('--out_file', type=str, default='fair',  help='The path to the output models.')
    parser.add_argument('--model_file', type=str, metavar='<model_file>', default='', help='The path to the model file, in case you want to continue training. (default value: '')')
    parser.add_argument('--n_pretrain_epoch', type=int, default=200, help='The number of epochs to pre-train the autoencoder. (default value: 100)')
    parser.add_argument('--n_epoch', type=int, default=2000, help='The number of epochs to train medGAN. (default value: 1000)')
    parser.add_argument('--n_discriminator_update', type=int, default=2, help='The number of times to update the discriminator per epoch. (default value: 2)')
    parser.add_argument('--n_generator_update', type=int, default=1, help='The number of times to update the generator per epoch. (default value: 1)')
    parser.add_argument('--pretrain_batch_size', type=int, default=100, help='The size of a single mini-batch for pre-training the autoencoder. (default value: 100)')
    parser.add_argument('--batch_size', type=int, default=100, help='The size of a single mini-batch for training medGAN. (default value: 1000)')
    parser.add_argument('--save_max_keep', type=int, default=0, help='The number of models to keep. Setting this to 0 will save models for every epoch. (default value: 0)')
    parser.add_argument('--generate_data', type=str2bool, default=False, help='If True the model generates data, if False the model is trained (default value: False)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parse_arguments(parser)

    data = np.load(args.data_file, allow_pickle = True)
    inputDim = data.shape[1]-1
    inputNum = data.shape[0]

    mg = Medgan(dataType=args.data_type,
                inputDim=inputDim,
                embeddingDim=args.embed_size,
                randomDim=args.noise_size,
                generatorDims=args.generator_size,
                discriminatorDims=args.discriminator_size,
                compressDims=args.compressor_size,
                decompressDims=args.decompressor_size,
                bnDecay=args.batchnorm_decay,
                l2scale=args.L2)

    # True for generation, False for training
    if not args.generate_data:
    # Training
        mg.train(dataPath=args.data_file,
                 modelPath=args.model_file,
                 outPath=args.out_file,
                 pretrainEpochs=args.n_pretrain_epoch,
                 nEpochs=args.n_epoch,
                 discriminatorTrainPeriod=args.n_discriminator_update,
                 generatorTrainPeriod=args.n_generator_update,
                 pretrainBatchSize=args.pretrain_batch_size,
                 batchSize=args.batch_size,
                 saveMaxKeep=args.save_max_keep)
    else:
    # Generate synthetic data using a trained model
    # You must specify "--model_file" and "<out_file>" to generate synthetic data.
        mg.generateData(nSamples=inputNum,
                        modelFile=args.model_file,
                        batchSize=args.batch_size,
                        outFile=args.out_file)
