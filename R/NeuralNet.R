#Nueral network 

#activation functions 

relu<-function(x)
{
		if (x<0) return (0) else return(x)
	
}
diff.relu<-function(x)
{
	
	if(x>0) {
		return(1)
	}else if(x==0){
		return(runif(1,0,1))
	}else{
		return(0)
	}
}
#sigmoid 
sigmoid<-function(x){
	if(x>35) x=35
	if(x<-35) x=-35
	return(1.0/1+exp(-x))
}

diff.sigmoid<-function(x)
{
	return(sigmoid(x)(1-sigmoid(x)))
}

## Main neural net class ## 
init<-function(minibatch_size,lrate)
{
	minbatch<<-minibatch_size
	lrate<<-lrate
	momentum <<-0
}
forwardpas<-function()
{
	
}
backprop<-function()
{
	
}
#' Stochastic minibatch gradient descendent
#' 
MSGD<-function(x_train,y_train){
	train_size = nrow(x_train)
	nsample <- sample(train_size,train_size,replace = F)
	it=0
	while(it<(train_size - minbatch))
	{
		x <- x_train[it:(minbatch+it),]
		y <- y_train[it:(minbatch+it)]
		#forward pass 
		z<- forwardpass(x)
		backprop(x,y,z)
		
	}
}



