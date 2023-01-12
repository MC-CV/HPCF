import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
from print_hook import PrintHook

class Recommender:
	def __init__(self, sess, handler):
		self.sess = sess
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				reses = self.testEpoch()
				log(self.makePrint('Test', ep, reses, test))
			if ep % args.tstEpoch == 0:
				self.saveHistory()
			print()
		reses = self.testEpoch()
		log(self.makePrint('Test', args.epoch, reses, True))
		self.saveHistory()

	def messagePropagate(self, lats, adj):
		return Activate(tf.sparse.sparse_dense_matmul(adj, lats), self.actFunc)

	def hyperPropagate(self, lats, adj):
		lat1 = Activate(tf.transpose(adj) @ lats, self.actFunc)
		lat2 = tf.transpose(FC(tf.transpose(lat1), args.hyperNum, activation=self.actFunc)) + lat1
		lat3 = tf.transpose(FC(tf.transpose(lat2), args.hyperNum, activation=self.actFunc)) + lat2
		lat4 = tf.transpose(FC(tf.transpose(lat3), args.hyperNum, activation=self.actFunc)) + lat3
		ret = Activate(adj @ lat4, self.actFunc)
		# ret = adj @ lat4
		return ret

	def edgeDropout(self, mat):
		def dropOneMat(mat):
			indices = mat.indices
			values = mat.values
			shape = mat.dense_shape
			# newVals = tf.to_float(tf.sign(tf.nn.dropout(values, self.keepRate)))
			newVals = tf.nn.dropout(values, self.keepRate)
			return tf.sparse.SparseTensor(indices, newVals, shape)
		return dropOneMat(mat)

	def ours(self):
		uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
		iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
		uhyper = NNs.defineParam('uhyper', [args.latdim, args.hyperNum], reg=True)
		ihyper = NNs.defineParam('ihyper', [args.latdim, args.hyperNum], reg=True)
		uuHyper = (uEmbed0 @ uhyper)
		iiHyper = (iEmbed0 @ ihyper)

		ulats = [uEmbed0]
		ilats = [iEmbed0]
		gnnULats = []
		gnnILats = []
		hyperULats = []
		hyperILats = []

		for i in range(args.gnn_layer):
			ulat = self.messagePropagate(ilats[-1], self.edgeDropout(self.adj))
			ilat = self.messagePropagate(ulats[-1], self.edgeDropout(self.tpAdj))

			hyperULat = self.hyperPropagate(ulats[-1], tf.nn.dropout(uuHyper, self.keepRate))
			hyperILat = self.hyperPropagate(ilats[-1], tf.nn.dropout(iiHyper, self.keepRate))
      
			user_shared_emb, user_noise_emb = self.denoise(hyperULat,ulat)
			item_shared_emb, item_noise_emb = self.denoise(hyperILat,ilat)
			# user_shared_emb2, user_noise_emb2 = self.denoise(ulat,hyperULat)
			# item_shared_emb2, item_noise_emb2 = self.denoise(ilat,hyperILat)
   
			gnnULats.append(ulat)
			gnnILats.append(ilat)
			hyperULats.append(hyperULat)
			hyperILats.append(hyperILat)
   
			ulat_layer_shared, ulat_layer_noise = self.denoise(ulats[-1], ulat)
			ilat_layer_shared, ilat_layer_noise = self.denoise(ilats[-1], ilat)

			if args.data == 'yelp':
				atten_local_user = FC(ulat, 2, reg=True, useBias=True,
								activation=self.actFunc, name='atten_local_user'+str(i), reuse=True)
				atten_local_item = FC(ilat, 2, reg=True, useBias=True,
								activation=self.actFunc, name='atten_local_item'+str(i), reuse=True)
			else:
				atten_local_user = FC(ulat, 2, reg=True, useBias=True,
								activation='softmax', name='atten_local_user'+str(i), reuse=True)
				atten_local_item = FC(ilat, 2, reg=True, useBias=True,
								activation='softmax', name='atten_local_item'+str(i), reuse=True)				
			# import pdb;pdb.set_trace()
			# ulats.append(ulat + tf.einsum('abc,ba->bc', tf.stack([user_shared_emb,user_noise_emb]), atten_lg_user) 
            #     + tf.einsum('abc,ba->bc', tf.stack([ulat_layer_shared,ulat_layer_noise]), atten_local_user))
			# ilats.append(ilat + tf.einsum('abc,ba->bc', tf.stack([item_shared_emb,item_noise_emb]), atten_lg_item) 
            #     + tf.einsum('abc,ba->bc', tf.stack([ilat_layer_shared,ilat_layer_noise]), atten_local_item)) 
			ulats.append(ulat + hyperULat + tf.einsum('abc,ba->bc', tf.stack([ulat_layer_shared,ulat_layer_noise]), atten_local_user))
			ilats.append(ilat + hyperILat + tf.einsum('abc,ba->bc', tf.stack([ilat_layer_shared,ilat_layer_noise]), atten_local_item))

		ulat = tf.add_n(ulats)
		ilat = tf.add_n(ilats)
		# ulat_global = tf.add_n(ulats_global)
		# ilat_global = tf.add_n(ilats_global)

		pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
		pckIlat = tf.nn.embedding_lookup(ilat, self.iids)
		preds = tf.reduce_sum(pckUlat * pckIlat, axis=-1)

		# pckUlat_global = tf.nn.embedding_lookup(ulat_global, self.uids)
		# pckIlat_global = tf.nn.embedding_lookup(ilat_global, self.iids)
		# preds_global = tf.reduce_sum(pckUlat_global * pckIlat_global, axis=-1)
		return preds, ulat, ilat

	def denoise(self, origin_emb, target_emb):
		res_array = tf.expand_dims(tf.reduce_sum(tf.multiply(origin_emb,target_emb),axis=1),-1)*target_emb
		norm_num = tf.norm(target_emb, axis=1)*tf.norm(target_emb, axis=1)+1e-12
		clear_emb = res_array/tf.expand_dims(norm_num,-1)
		noise_emb = origin_emb - clear_emb
		if False:
			a = tf.cast(tf.reduce_sum(tf.multiply(origin_emb,target_emb),axis=1)>=0, tf.float32)
			clear_emb *= tf.expand_dims(a,-1)
		# return clear_emb*0.1, noise_emb*0.1
		return clear_emb, noise_emb


	def tstPred(self, ulat, ilat):
		pckUlat = tf.nn.embedding_lookup(ulat, self.uids)
		allPreds = pckUlat @ tf.transpose(ilat)
		allPreds = allPreds * (1 - self.trnPosMask) - self.trnPosMask * 1e8
		vals, locs = tf.nn.top_k(allPreds, args.shoot)
		vals2, locs2 = tf.nn.top_k(allPreds, 2*args.shoot)
		return locs, locs2
		# return locs

	def prepareModel(self):
		self.keepRate = tf.placeholder(dtype=tf.float32, shape=[])
		NNs.leaky = args.leaky
		self.actFunc = 'leakyRelu'
		adj = self.handler.trnMat
		idx, data, shape = transToLsts(adj, norm=True)
		self.adj = tf.sparse.SparseTensor(idx, data, shape)
		idx, data, shape = transToLsts(transpose(adj), norm=True)
		self.tpAdj = tf.sparse.SparseTensor(idx, data, shape)
		self.uids = tf.placeholder(name='uids', dtype=tf.int32, shape=[None])
		self.iids = tf.placeholder(name='iids', dtype=tf.int32, shape=[None])
		self.trnPosMask = tf.placeholder(name='trnPosMask', dtype=tf.float32, shape=[None, args.item])

		# preds, preds_global, sslloss, ulat, ilat, ulat_global, ilat_global = self.ours()
		self.preds, ulat, ilat = self.ours()
		# self.preds = args.alpha*preds+(1-args.alpha)*preds_global
		# self.preds = preds
		# ulat_final = args.alpha*ulat+args.beta*ulat_global
		# ilat_final = args.alpha*ilat+args.beta*ilat_global
		self.topLocs, self.topLocs2 = self.tstPred(ulat, ilat)
		# self.topLocs = self.tstPred(ulat, ilat)
		# self.topLocs = args.alpha*self.tstPred(ulat, ilat)+args.beta*self.tstPred(ulat_global, ilat_global)
		# self.topLocs = self.tstPred(ulat_global, ilat_global)

		sampNum = tf.shape(self.uids)[0] // 2
		posPred = tf.slice(self.preds, [0], [sampNum])
		negPred = tf.slice(self.preds, [sampNum], [-1])
		# posPred2 = tf.slice(preds_global, [0], [sampNum])
		# negPred2 = tf.slice(preds_global, [sampNum], [-1])

		self.preLoss = tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
		# self.preLoss += args.beta*tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred2 - negPred2))) / args.batch
		self.regLoss = args.reg * Regularize()
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def sampleTrainBatch(self, batIds, labelMat):
		temLabel = labelMat[batIds].toarray()
		batch = len(batIds)
		temlen = batch * 2 * args.sampNum
		uLocs = [None] * temlen
		iLocs = [None] * temlen
		cur = 0
		for i in range(batch):
			posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
			sampNum = min(args.sampNum, len(posset))
			if sampNum == 0:
				poslocs = [np.random.choice(args.item)]
				neglocs = [poslocs[0]]
			else:
				poslocs = np.random.choice(posset, sampNum)
				neglocs = negSamp(temLabel[i], sampNum, args.item)
			for j in range(sampNum):
				posloc = poslocs[j]
				negloc = neglocs[j]
				uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
				iLocs[cur] = posloc
				iLocs[cur+temlen//2] = negloc
				cur += 1
		uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
		iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
		return uLocs, iLocs

	def trainEpoch(self):
		num = args.user
		sfIds = np.random.permutation(num)[:args.trnNum]
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batIds = sfIds[st: ed]

			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			feed_dict = {}
			uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)
			feed_dict[self.uids] = uLocs
			feed_dict[self.iids] = iLocs
			feed_dict[self.keepRate] = args.keepRate

			res = self.sess.run(target, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			# log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def testEpoch(self):
		epochRecall, epochNdcg = [0] * 2
		epochRecall2, epochNdcg2 = [0] * 2
		ids = self.handler.tstUsrs
		num = len(ids)
		tstBat = args.batch
		steps = int(np.ceil(num / tstBat))
		tstNum = 0
		for i in range(steps):
			st = i * tstBat
			ed = min((i+1) * tstBat, num)
			batIds = ids[st: ed]
			feed_dict = {}

			trnPosMask = self.handler.trnMat[batIds].toarray()
			feed_dict[self.uids] = batIds
			feed_dict[self.trnPosMask] = trnPosMask
			feed_dict[self.keepRate] = 1.0
			topLocs = self.sess.run(self.topLocs, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			topLocs2 = self.sess.run(self.topLocs2, feed_dict=feed_dict, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			recall, ndcg = self.calcRes(topLocs, self.handler.tstLocs, batIds, args.shoot)
			recall2, ndcg2 = self.calcRes(topLocs2, self.handler.tstLocs, batIds, args.shoot*2)
   
			epochRecall += recall
			epochNdcg += ndcg
			epochRecall2 += recall2
			epochNdcg2 += ndcg2
			# log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
		ret = dict()
		ret['Recall20'] = epochRecall / num
		ret['NDCG20'] = epochNdcg / num
		ret['Recall40'] = epochRecall2 / num
		ret['NDCG40'] = epochNdcg2 / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds, shoot):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		recallBig = 0
		ndcgBig =0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, shoot))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	import random
	random.seed(42)  # 为python设置随机种子
	np.random.seed(42)  # 为numpy设置随机种子
	tf.set_random_seed(42)  # tf cpu fix seed
	os.environ['TF_DETERMINISTIC_OPS'] = '1'
	os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
	log_dir = 'log/'+ args.data + '/' + os.path.basename(__file__)

	if not os.path.isdir(log_dir):
		os.makedirs(log_dir)
	import datetime

	log_file = open(log_dir + '/log' + str(datetime.datetime.now()), 'w')


	def my_hook_out(text):
		log_file.write(text)
		log_file.flush()
		return 1, 0, text

	ph_out = PrintHook()
	ph_out.Start(my_hook_out)
	print('Use gpu id:', args.gpu)
	for arg in vars(args):
		print(arg + '=' + str(getattr(args, arg)))
  
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, handler)
		recom.run()
