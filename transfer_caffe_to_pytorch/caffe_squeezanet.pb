
�
fire2/expand1x1Convfire2/relu_squeeze1x1"
pads

        "
group"
kernel_shape
@"
strides
"
use_bias("/
_output_shapes
:���������88@
�
fire5/squeeze1x1Convfire4/concat"/
_output_shapes
:��������� "
pads

        "
group"
kernel_shape	
� "
strides
"
use_bias(
^
fire2/relu_expand3x3Relufire2/expand3x3"/
_output_shapes
:���������88@
`
fire6/relu_squeeze1x1Relufire6/squeeze1x1"/
_output_shapes
:���������0
�
fire3/expand3x3Convfire3/relu_squeeze1x1"
kernel_shape
@"
strides
"
use_bias("/
_output_shapes
:���������88@"
pads

    "
group
`
fire5/relu_squeeze1x1Relufire5/squeeze1x1"/
_output_shapes
:��������� 
�
fire7/expand1x1Convfire7/relu_squeeze1x1"
pads

        "
group"
kernel_shape	
0�"
strides
"
use_bias("0
_output_shapes
:����������
�
fire4/concatConcatfire4/relu_expand1x1fire4/relu_expand3x3"0
_output_shapes
:����������"

axis
�
fire4/expand1x1Convfire4/relu_squeeze1x1"
strides
"
use_bias("0
_output_shapes
:����������"
pads

        "
group"
kernel_shape	
 �
�
fire6/concatConcatfire6/relu_expand1x1fire6/relu_expand3x3"0
_output_shapes
:����������"

axis
�
fire2/concatConcatfire2/relu_expand1x1fire2/relu_expand3x3"

axis"0
_output_shapes
:���������88�
_
fire5/relu_expand3x3Relufire5/expand3x3"0
_output_shapes
:����������
�
fire6/expand1x1Convfire6/relu_squeeze1x1"0
_output_shapes
:����������"
pads

        "
group"
kernel_shape	
0�"
strides
"
use_bias(
�
fire3/concatConcatfire3/relu_expand1x1fire3/relu_expand3x3"

axis"0
_output_shapes
:���������88�
_
fire4/relu_expand1x1Relufire4/expand1x1"0
_output_shapes
:����������
`
fire7/relu_squeeze1x1Relufire7/squeeze1x1"/
_output_shapes
:���������0
_
fire6/relu_expand3x3Relufire6/expand3x3"0
_output_shapes
:����������
�
my-conv10-teslaConvdrop9"
kernel_shape	
�	"
strides
"
use_bias("/
_output_shapes
:���������	"
pads

        "
group
^
fire2/relu_expand1x1Relufire2/expand1x1"/
_output_shapes
:���������88@
�
fire6/expand3x3Convfire6/relu_squeeze1x1"
kernel_shape	
0�"
strides
"
use_bias("0
_output_shapes
:����������"
pads

    "
group
�
fire3/expand1x1Convfire3/relu_squeeze1x1"/
_output_shapes
:���������88@"
pads

        "
group"
kernel_shape
@"
strides
"
use_bias(
_
fire7/relu_expand1x1Relufire7/expand1x1"0
_output_shapes
:����������
�
fire8/squeeze1x1Convfire7/concat"
group"
kernel_shape	
�@"
strides
"
use_bias("/
_output_shapes
:���������@"
pads

        
`
fire4/relu_squeeze1x1Relufire4/squeeze1x1"/
_output_shapes
:��������� 
�
fire9/expand1x1Convfire9/relu_squeeze1x1"0
_output_shapes
:����������"
pads

        "
group"
kernel_shape	
@�"
strides
"
use_bias(
d
drop9Dropoutfire9/concat"
	keep_prob%   ?"0
_output_shapes
:����������
�
conv1Convdata"
kernel_shape
@"
strides
"
use_bias("/
_output_shapes
:���������qq@"
pads

      "
group
[
heatmapConcatrelu_conv10"

axis"/
_output_shapes
:���������	
_
fire8/relu_expand1x1Relufire8/expand1x1"0
_output_shapes
:����������
�
fire8/concatConcatfire8/relu_expand1x1fire8/relu_expand3x3"0
_output_shapes
:����������"

axis
�
pool3Poolfire3/concat"
kernel_shape
"
pooling_typeMAX"
strides
"0
_output_shapes
:����������"
pads

      
�
pool1Pool
relu_conv1"/
_output_shapes
:���������88@"
pads

      "
kernel_shape
"
pooling_typeMAX"
strides

H
probSoftmaxpool10"/
_output_shapes
:���������	
�
pool5Poolfire5/concat"
kernel_shape
"
pooling_typeMAX"
strides
"0
_output_shapes
:����������"
pads

      
�
fire6/squeeze1x1Convpool5"/
_output_shapes
:���������0"
pads

        "
group"
kernel_shape	
�0"
strides
"
use_bias(
�
fire5/expand3x3Convfire5/relu_squeeze1x1"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape	
 �"
strides
"
use_bias(
_
fire7/relu_expand3x3Relufire7/expand3x3"0
_output_shapes
:����������
�
fire7/squeeze1x1Convfire6/concat"/
_output_shapes
:���������0"
pads

        "
group"
kernel_shape	
�0"
strides
"
use_bias(
�
pool10Poolrelu_conv10"
pads

        "
kernel_shape
"
pooling_typeAVG"
strides
"/
_output_shapes
:���������	
�
fire9/expand3x3Convfire9/relu_squeeze1x1"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape	
@�"
strides
"
use_bias(
�
fire5/concatConcatfire5/relu_expand1x1fire5/relu_expand3x3"

axis"0
_output_shapes
:����������
_
fire4/relu_expand3x3Relufire4/expand3x3"0
_output_shapes
:����������
�
fire8/expand1x1Convfire8/relu_squeeze1x1"
kernel_shape	
@�"
strides
"
use_bias("0
_output_shapes
:����������"
pads

        "
group
_
fire8/relu_expand3x3Relufire8/expand3x3"0
_output_shapes
:����������
_
fire9/relu_expand3x3Relufire9/expand3x3"0
_output_shapes
:����������
l
data	DataInput"1
_output_shapes
:�����������"&
shape:�����������
`
fire3/relu_squeeze1x1Relufire3/squeeze1x1"/
_output_shapes
:���������88
_
fire9/relu_expand1x1Relufire9/expand1x1"0
_output_shapes
:����������
�
fire2/expand3x3Convfire2/relu_squeeze1x1"
strides
"
use_bias("/
_output_shapes
:���������88@"
pads

    "
group"
kernel_shape
@
_
fire5/relu_expand1x1Relufire5/expand1x1"0
_output_shapes
:����������
�
fire9/concatConcatfire9/relu_expand1x1fire9/relu_expand3x3"0
_output_shapes
:����������"

axis
�
fire2/squeeze1x1Convpool1"/
_output_shapes
:���������88"
pads

        "
group"
kernel_shape
@"
strides
"
use_bias(
J

relu_conv1Reluconv1"/
_output_shapes
:���������qq@
�
fire4/squeeze1x1Convpool3"
use_bias("/
_output_shapes
:��������� "
pads

        "
group"
kernel_shape	
� "
strides

�
fire7/expand3x3Convfire7/relu_squeeze1x1"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape	
0�"
strides
"
use_bias(
�
fire5/expand1x1Convfire5/relu_squeeze1x1"
pads

        "
group"
kernel_shape	
 �"
strides
"
use_bias("0
_output_shapes
:����������
^
fire3/relu_expand1x1Relufire3/expand1x1"/
_output_shapes
:���������88@
�
fire3/squeeze1x1Convfire2/concat"/
_output_shapes
:���������88"
pads

        "
group"
kernel_shape	
�"
strides
"
use_bias(
�
fire8/expand3x3Convfire8/relu_squeeze1x1"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape	
@�"
strides
"
use_bias(
`
fire9/relu_squeeze1x1Relufire9/squeeze1x1"/
_output_shapes
:���������@
_
fire6/relu_expand1x1Relufire6/expand1x1"0
_output_shapes
:����������
`
fire8/relu_squeeze1x1Relufire8/squeeze1x1"/
_output_shapes
:���������@
�
fire7/concatConcatfire7/relu_expand1x1fire7/relu_expand3x3"

axis"0
_output_shapes
:����������
U
relu_conv10Relumy-conv10-tesla"/
_output_shapes
:���������	
�
fire9/squeeze1x1Convfire8/concat"
kernel_shape	
�@"
strides
"
use_bias("/
_output_shapes
:���������@"
pads

        "
group
^
fire3/relu_expand3x3Relufire3/expand3x3"/
_output_shapes
:���������88@
�
fire4/expand3x3Convfire4/relu_squeeze1x1"0
_output_shapes
:����������"
pads

    "
group"
kernel_shape	
 �"
strides
"
use_bias(
`
fire2/relu_squeeze1x1Relufire2/squeeze1x1"/
_output_shapes
:���������88