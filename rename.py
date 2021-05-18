import os

for imgfile in os.listdir('./img'):
    newfilename = imgfile.replace(' (','_').replace(').jpg','.jpg')
    os.rename(r'./img/'+imgfile,r'./img/'+newfilename)
