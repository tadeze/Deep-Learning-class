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






