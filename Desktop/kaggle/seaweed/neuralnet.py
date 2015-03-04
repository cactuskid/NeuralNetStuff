from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neural_network import BernoulliRBM
from sklearn.preprocessing import MultiLabelBinarizer

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    Parameters
    ----------
    y_true : array, shape = [n_samples]
            true class, intergers in [0, n_classes - 1)
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss
 



class neuralNet:
	
	def __init__(self,numLayers,n_components,learning_rates,featureGroups, trainY, X,Y):
		
		#dictionaries for learning rates
		self.learning_rates=learning_rates

		#dictionaries for number of components
		self.n_components=n_components

		#top + label layer
		self.numLayers = numLayers

		#dictionary of numpy vectors with Feature groups to learn from in bottom layers
		#featuregroups[layer][groupnumber]
		self.featureGroups = featureGroups

		self.trainY = trainY

		#dic to store rbms
		self.Layers={}
		self.output = {}


		print "Training RBM layers"
		#hyper params
		self.n = n_components
		self.rate =learning_rate1
		
		#train lower layers
		#train with blocks of features
		#overlap is ok...

		for i in range(numBottomLayers):
			self.bottomlayers[i] = {}

			columns =0
			for featureGroup in featureGroups[i]:
				columns += featureGroup.shape[0]
			newX = np.zeros((X.shape[0], columns))

			for featureGroup in featureGroups[i]:
				print 'making rbm layer ' + str(i) 
				rbm = BernoulliRBM(n_components=self.n_components[i] , learning_rate=self.learning_rates[i], batch_size=10, n_iter=10, verbose=1, random_state=0)		
				if trainY[i]:
					rbm.fit(X[featureGroup,:],Y)
				if not trainY[i]:	
					rbm.fit(X[featureGroup,:])
				self.layers[i][featureGroup] = rbm
				out = rbm2.transform(X[featureGroup,:])
				newX[ n_components * i : n_components*(i+1) - 1 , : ] = out
				self.output[i][featureGroup] = out
				print 'layer done'
				#TODO concatenate the new X
			X = newX

		

	def backPropagate (self,ErrorVec, outputs, lastX,neural_output , M,N):


    output_deltas = [0.0] * self.no
   	

    #final layer
    i = self.numLayers -1
    featureGroup = self.featureGroups[i][0]


	Actual = MultiLabelBinarizer().fit_transform(Y)
	ErrorVec = Actual - self.output[i][featureGroup]

    for k in range(self.no):
    	#calculate delta based on output of final layer
      output_deltas[k] =  ErrorVec[k] * sigDeriv( self.output[i][featureGroup][k] ) 
   
   
   #TODO propagate on final layer
   




   #TODO loop here on layers,featuregroups
   hidden = self.bottomlayers[i].intercept_hidden_
   visible = self.bottomlayers[i].intercept_visible_
   weights = self.bottomlayers[i].components_
   featureGroup = self.featureGroups[i]
   


    # update output weights
    for j in range(self.nh):
      for k in range(self.no):
        # output_deltas[k] * self.ah[j] is the full derivative of dError/dweight[j][k]
        change = output_deltas[k] * self.ah[j]
        self.wo[j][k] += N*change + M*self.co[j][k]
        self.co[j][k] = change

    # calc hidden deltas
    hidden_deltas = [0.0] * self.nh
    for j in range(self.nh):
      error = 0.0
      for k in range(self.no):
        error += output_deltas[k] * self.wo[j][k]
      hidden_deltas[j] = error * sigDeriv(self.ah[j])
    
    #update input weights
    for i in range (self.ni):
      for j in range (self.nh):
        change = hidden_deltas[j] * self.ai[i]
        #print 'activation',self.ai[i],'synapse',i,j,'change',change
        #use delta version of backprop to increase learning speed, M is momentum term
        self.wi[i][j] += N*change + M*self.ci[i][j]
        self.ci[i][j] = change
    

    # calc combined error
    # 1/2 for differential convenience & **2 for modulus
    error = 0.0
    for k in range(len(targets)):
      error = 0.5 * (targets[k]-self.ao[k])**2
    return error


	def sigmoid(in):
	    f = 1 /( 1 + math.exp(-in))
	    return f

	def sigDeriv(in):
	    f = sigmoid(in) * (1-sigmoid(in))
	    return f

	def retrain(self,X,Y, learning):

		for i in range(self.numBottomLayers):
			columns =0
			for featureGroup in self.featuregroups[i]:
				columns += featureGroup.shape[0]
			newX = np.zeros((X.shape[0], columns))
			for featureGroup in self.featureGroups[i]:
				print ' retraining rbm layer ' + str(i) 
				rbm = bottomlayers[i][featureGroup]
				#adapt learning rate to error 
				rbm.learning_rate = self.learning_rates[i]
				if self.trainY[i]
					rbm.partial_fit(X[featureGroup,:])
				if not self.trainY[i]
					rbm.fit(X[featureGroup,:],Y)
				out = rbm.transform(X[featureGroup,:])
				newX[ n_components * i : n_components*(i+1) - 1 , : ] = out
				self.output[i][featureGroup] = out
				print 'layer done'
				#TODO concatenate the new X
			X = newX


	def TransformData(self, X):
		
		outputs= {}
		n_components = self.n_components1
		for i in range(self.numBottomLayers):
			outputs[i] = {}
			newX = np.zeros((,))
			for featureGroup in self.featuregroups:
				rbm = self.bottomlayers[i][featureGroup]
				out = rbm.transform(X[featureGroup,:])
				newX[ n_components * i : n_components*(i+1) - 1 , : ] = out
				print 'transform bottom layer ' + str(i) 
				self.output[i][featureGroup] = out
			X = newX


	#implement backprop error using RF classifier probas for each category
	def multiClass_backprop(self, X,Y,M,N):
		

		Actual = MultiLabelBinarizer().fit_transform(Y)
		
		self.TransformData(X)
		#transform the class into a one-hot representation of the target values for the neurons
		self.backPropagate( Actual , M,N)
		

