# -*- coding: utf-8 -*-
""" Created on 20/01/2022 by Bhakti Chotalia """

#importing tensorflow library
import tensorflow as tf

#declaring matrix 2 x 3
t1=tf.constant([1,3,5,7,9,11],shape=[2,3])
print(t1)

#declaring matrix 3 x 2
t2=tf.constant([2,4,6,8,10,12],shape=[3,2])
print("\n",t2)

#multiplying t1 and t2 matrix using matmul funtion
t3=tf.matmul(t1,t2)
print("\nProduct:",t3)

#generating Eigen matrix of 2 x 2 using random number
e_matrix_A=tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32,name="matrixA")
print("\nMatrix A:\n{}\n\n".format(e_matrix_A))

#calculating Eigen values and Eigen vectors using linalg function
eigen_values_A,eigen_vectors_A=tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors:\n{}\n\nEigen Values:\n{}\n".format(eigen_vectors_A,eigen_values_A))
