import struct
import imghdr
import os


def get_image_size(fname):
    with open(fname, 'rb') as fhandle:
        if imghdr.what(fname) == 'jpeg':
            try:
                fhandle.seek(0) # Read 0xff next
                size = 2
                ftype = 0
                while not 0xc0 <= ftype <= 0xcf:
                    fhandle.seek(size, 1)
                    byte = fhandle.read(1)
                    while ord(byte) == 0xff:
                        byte = fhandle.read(1)
                    ftype = ord(byte)
                    size = struct.unpack('>H', fhandle.read(2))[0] - 2
                # We are at a SOFn block
                fhandle.seek(1, 1)  # Skip `precision' byte.
                height, width = struct.unpack('>HH', fhandle.read(4))
            except Exception: #IGNORE:W0703
                return
        else:
            return
        return width, height

for txtfile in os.listdir('./newtxt'):
    img_width , img_height = get_image_size('./img/'+txtfile.replace('.txt','.jpg'))
    print("filename: {:} || {:}x{:} ".format(txtfile,img_width,img_height))

    with open('./newtxt/'+txtfile,'r') as fr:
        with open('./newtxt_1/'+txtfile,'a') as fw:
            lines = fr.readlines()
            for line in lines:
                nums = line.split(' ')
                nums = list(filter(None,nums))

                label_center_x = (int(nums[1])+int(nums[3]))/2/img_width
                label_center_y = (int(nums[2])+int(nums[4]))/2/img_height
                label_width = (int(nums[3])-int(nums[1]))/img_width
                label_height = (int(nums[4])-int(nums[2]))/img_width

                nums_write = '{:} {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(0,label_center_x,label_center_y,label_width,label_height)

                print(nums_write)
                fw.write(nums_write)