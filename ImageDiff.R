library(ANTsR)

readData <- function(address,patientId,imageType,iteration,modality,docId)
{
  if(imageType == "masks")
  {
    #Reading masks
    finalAd = paste0(address,"/training0",toString(patientId),"/masks/","training0",toString(patientId),"_0",toString(iteration),"_mask",toString(docId),".nii")
    
  }
  else if (imageType == "preprocessed")
  {
    finalAd = paste0(address,"/training0",toString(patientId),"/preprocessed/","training0",toString(patientId),"_0",toString(iteration),"_",toString(modality),"_pp.nii")
    print(finalAd)
  }
  else
  {
    # for reading orig images
  }
  return(antsImageRead(finalAd,3,"float"))
}


# write diff images into output files
writeData <- function(image,address,patientId,iteration1,iteration2,modality)
{
  
  finalAd <- paste0(address,"/",toString(patientId))
  
  finalAd <- paste0(finalAd,"/",modality,"/",toString(patientId),"_",toString(iteration1),toString(iteration2),modality,".nii")
  #print(finalAd)
  antsImageWrite(image,finalAd)
  
  
}

source('/home/roshan/Documents/Acads/GKProject/Codes/readData.R')
source('/home/roshan/Documents/Acads/GKProject/Codes/writeData.R')


# directories
inputAdd = "/home/roshan/Documents/Acads/GKProject/DATA/training"
outputAdd = "/home/roshan/Documents/Acads/GKProject/DATA/training"

#initializinf output directory folders
for(i in 1:4)
{
  finalAd <- paste0(outputAdd,"/",toString(i))
  if(!file.exists(finalAd))
  {
    print("File doesnt exist")
    dir.create(finalAd)
    for(mode in c("flair","mprage","pd","t2"))
    {
      dir.create( paste0(finalAd,"/",mode) )
      
    }
  }
}

##patient 1

ID<- 1
mode <- "flair"
for(mode in c("flair","mprage","pd","t2"))
{
  img =list()

  for(i in 1:4)
  {
    temp = readData(preAdd,1,"preprocessed",i,mode,0)
    img = append(img,temp)
  }

  for(i in 1:4)
  {
    for(j in 1:4)
    {
      if(i!=j)
      {
        diff = img[i][[1]]-img[j][[1]]
        writeData(image=diff,address = outputAdd,patientId = ID,iteration1 = i,iteration2 = j, modality = mode)   
      }
    }
  }
}

