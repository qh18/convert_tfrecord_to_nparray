#--------------------------------------------------------------#
# This code is an example to read Youtube-8m frame-level data,
# convert them into numpy arrays, and save them in gzip format  
#

import tensorflow as tf
from tensorflow.python.ops import parsing_ops
from tensorflow.contrib.slim.python.slim.data import parallel_reader
import numpy as np
import cPickle as pickle  
import gzip


# from  youtube-8m/utils.py
def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias


def qh_resize_axis(feats, value=0, dim=300):
    dim0, dim1 = feats.shape
    temp_b = np.zeros((dim-dim0, dim1))
    feats = np.concatenate((feats,temp_b),axis=0)
    return feats
    
def main():
  file_path = 'your_data_path/' + '*.tfrecord'
  max_quantized_value = 2
  min_quantized_value = -2

  reader = tf.TFRecordReader
  data_sources = file_path
  _, data = parallel_reader.parallel_read(
      data_sources,
      reader_class=reader,
      num_epochs=1,
      num_readers=1,
      shuffle=False,
      capacity=256,
      min_after_dequeue=1)

  context_features, sequence_features = parsing_ops.parse_single_sequence_example(data, context_features={
      'video_id': tf.VarLenFeature(tf.string),
      'labels': tf.VarLenFeature(tf.int64),
    }, sequence_features={
      'rgb': tf.FixedLenSequenceFeature([], tf.string),
      'audio':  tf.FixedLenSequenceFeature([], tf.string),  
    }, example_name="")

  with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    total_av_feas = []

    try:
      while not coord.should_stop():
        meta, av_feas = sess.run((context_features, sequence_features))

        vid = meta['video_id'].values[0]
        labels = meta['labels'].values

        labels_boolean = sess.run(tf.sparse_to_dense(context_features['labels'].values, (4716,), 1, validate_indices=False))

        inc3_fea = av_feas['rgb']
        ad3_fea = av_feas['audio']

        img_feas = []
        for ii in range(len(inc3_fea)):
            v = np.fromstring(inc3_fea[ii], dtype=np.uint8)
            img_feas.append(v)
        img_feas = np.vstack(img_feas)
        img_feas = Dequantize(img_feas,max_quantized_value, min_quantized_value)
        img_feas = qh_resize_axis(img_feas, 0, 300) # zero padding
 
        aud_feas = []
        for ii in range(len(ad3_fea)):
            v = np.fromstring(ad3_fea[ii], dtype=np.uint8)
            aud_feas.append(v)
        aud_feas = np.vstack(aud_feas)
        aud_feas = Dequantize(aud_feas,max_quantized_value, min_quantized_value)
        aud_feas = qh_resize_axis(aud_feas, 0, 300) # zero padding

        #--- concatenate img and audio features ---#
        av_feats = np.concatenate((img_feas,aud_feas), axis=1)
        temp_dict = {"vid": vid, "label": labels_boolean, "va_1024_128_300d": av_feats}
        total_av_feas.append(temp_dict)
        print vid
        
    except tf.errors.OutOfRangeError:
      print('Finished extracting.')
    finally:
      L = len(total_av_feas)
      N = 2000
      if (L > N):
          new_filename = 'your_output_path/yt8m_framelevel' + str(index) + '.pgz'
          with gzip.GzipFile(new_filename, 'wb') as fp:
              pickle.dump(total_feas[:N], fp)
          fp.close()
          total_av_feas = total_av_feas[N:L]
          index = index + 1
          
      coord.request_stop()
      coord.join(threads)

  sess.close()

if __name__ == '__main__':
  main()
