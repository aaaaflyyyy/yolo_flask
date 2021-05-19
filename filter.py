import os

for txtfile in os.listdir('./txt'):
    with open('./txt/'+txtfile,'r') as fr:
        lines = fr.readlines()
        for line in lines:
            nums = line.split(' ')
            nums = list(filter(None,nums))
            if(nums[0] != '0'):
                continue
            else:                
                nums_write = ''
                for i,num in enumerate(nums):
                    nums_write += num
                    
                    if i < len(nums)-1:
                        nums_write +=' '

                with open('./newtxt/'+txtfile,'a') as fw:
                    fw.write(nums_write)
