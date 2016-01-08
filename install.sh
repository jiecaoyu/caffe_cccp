make clean
make CXX=/usr/bin/g++-4.6 -j8
make pycaffe
protoc src/caffe/proto/caffe.proto --cpp_out=.
mkdir include/caffe/proto
mv src/caffe/proto/caffe.pb.h include/caffe/proto
echo "Finish installing."
