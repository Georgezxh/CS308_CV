import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1
  #filter /= np.sum(filter)
  
  f = filter.shape[0]
  g = filter.shape[1]
  hf = (f-1)/2
  hf = int(hf)
  hg = (g-1)/2
  hg = int(hg)
  
  c = image.shape[2]

  if(c==1):
    image = np.pad(image, ((hf, hf),(hg, hg) ),  'constant', constant_values=(0,0))
  else:
  #######padding操作填充图像周围，避免产生四周没有被滤波的空隙
    channel_one = image[:,:,0]
    channel_two = image[:,:,1]
    channel_three = image[:,:,2]
    
    channel_one = np.pad(channel_one, ((hf, hf),(hg, hg) ),  'constant', constant_values=(0,0))
    channel_two = np.pad(channel_two, ((hf, hf),(hg, hg) ),  'constant', constant_values=(0,0))
    channel_three = np.pad(channel_three, ((hf, hf),(hg, hg) ),  'constant', constant_values=(0,0))
    
    image = np.dstack((channel_one,channel_two,channel_three))
  #Ack: https://blog.csdn.net/qq_37053885/article/details/80774703

  m = image.shape[0]
  n = image.shape[1]


  filtered_image = image.copy()

  for i in range(hf,m-hf):
    for j in range(hg,n-hg):
      for k in range(c):
        sum = 0
        for s in range(g):
          for z in range(f):
            sum += filter[z][s]*image[i-hf+z][j-hg+s][k]
        filtered_image[i][j][k] = sum

  m = m - hf
  n = n - hg
  return filtered_image[hf:m, hg:n, :]



  ### END OF STUDENT CODE ####
  ############################

  
def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
  #高斯滤波器留下低频
  low_frequencies = my_imfilter(image1,filter)

  high_frequencies = image2 - my_imfilter(image2,filter) 
  high_frequencies -= high_frequencies.min()
  high_frequencies = high_frequencies / high_frequencies.max()

  hybrid_image = low_frequencies + high_frequencies
  hybrid_image = hybrid_image/hybrid_image.max()

  return low_frequencies, high_frequencies, hybrid_image


  ### END OF STUDENT CODE ####
  ############################

  
