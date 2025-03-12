为什么要用pack_padded_sequence
在使用深度学习特别是LSTM进行文本分析时，经常会遇到文本长度不一样的情况，此时就需要对同一个batch中的不同文本使用padding的方式进行文本长度对齐，方便将训练数据输入到LSTM模型进行训练，同时为了保证模型训练的精度，应该同时告诉LSTM相关padding的情况，此时，pytorch中的pack_padded_sequence就有了用武之地。
直接从文本处理开始
假设我们有如下四段文字：
1.To the world you may be one person, but to one person you may be the world. 
2.No man or woman is worth your tears, and the one who is, won’t make you cry. 
3.Never frown, even when you are sad, because you never know who is falling in love with your smile. 
4.We met at the wrong time, but separated at the right time. The most urgent is to take the most beautiful scenery; the deepest wound was the most real emotions.
1234
将文本存储为test.txt，使用如下脚本转换成padding之后的矩阵：
import numpy as np
import wordfreq

vocab = {}
token_id = 1
lengths = []

with open('test.txt', 'r') as f:
    for l in f:
        tokens = wordfreq.tokenize(l.strip(), 'en')
        lengths.append(len(  ))
        for t in tokens:
            if t not in vocab:
                vocab[t] = token_id
                token_id += 1

x = np.zeros((len(lengths), max(lengths)))
l_no = 0
with open('test.txt', 'r') as f:
    for l in f:
        tokens = wordfreq.tokenize(l.strip(), 'en')
        for i in range(len(tokens)):
            x[l_no, i] = vocab[tokens[i]]
        l_no += 1
123456789101112131415161718192021222324
我们可以看到文本已经被转换成token_id矩阵，其中0为padding值。

使用pack_padded_sequence
接下来就是需要用到pack_padded_sequence的地方：
import torch
import torch.nn as nn
from torch.autograd import Variable

x = Variable(x)
lengths = torch.Tensor(lengths)
_, idx_sort = torch.sort(torch.Tensor(lengths), dim=0, descending=True)
_, idx_unsort = torch.sort(idx_sort, dim=0)

x = x.index_select(0, idx_sort)
lengths = list(lengths[idx_sort])
x_packed = nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
123456789101112
需要注意的是，pack_padded_sequence函数的参数，lengths需要从大到小排序，x为已根据长度大小排好序，batch_first如果设置为true，则x的第一维为batch_size，第二维为seq_length，否则相反。
打印x_packed如下：

可以看到，x的前二维已被合并成一维，同时原来x中的padding值0已经被消除，多出的batch_sizes可以看成是原来x中第二维即seq_length在第一维即batch_size中不为padding值的个数。
使用pad_packed_sequence
那么问题来了，x_packed经后续的LSTM处理之后，如何转换回padding形式呢？没错，这就是pad_packed_sequence的用处。
假设x_packed经LSTM网络输出后仍为x_packed（注：一般情况下，经LSTM网络输出应该有第三维，但方便起见，x_packed的第三维的维度可看成是1），则相应转换如下：
x_padded = nn.utils.rnn.pad_packed_sequence(x_packed, batch_first=True)
output = x_padded[0].index_select(0, idx_unsort)
12
需要注意的是，idx_unsort的作用在于将batch中的序列调整为原来的顺序。
打印output：

可以看出，与原来的x一样。
PackedSequence的用处
其实很简单，当之前的x_packed需要经过dropout等函数处理时，需要传入的是x_packed.data，是一个tensor，经过处理后，要将其重新封装成PackedSequence，再传入LSTM网络，示例如下：
dropout_output = nn.functional.dropout(x_packed.data, p=0.6, training=True)
x_dropout = nn.utils.rnn.PackedSequence(dropout_output, x_packed.batch_sizes)
12
参考
一个更直观的解释，来自博客
为什么有pad和pack操作？
先看一个例子，这个batch中有5个sample

如果不用pack和pad操作会有一个问题，什么问题呢？比如上图，句子“Yes”只有一个单词，但是padding了多余的pad符号，这样会导致LSTM对它的表示通过了非常多无用的字符，这样得到的句子表示就会有误差，更直观的如下图：

那么我们正确的做法应该是怎么样呢？
在上面这个例子，我们想要得到的表示仅仅是LSTM过完单词"Yes"之后的表示，而不是通过了多个无用的“Pad”得到的表示：如下图：

torch.nn.utils.rnn.pack_padded_sequence()
这里的pack，理解成压紧比较好。 将一个 填充过的变长序列 压紧。（填充时候，会有冗余，所以压紧一下）
其中pack的过程为：（注意pack的形式，不是按行压，而是按列压）

（下面方框内为PackedSequence对象，由data和batch_sizes组成）
pack之后，原来填充的 PAD（一般初始化为0）占位符被删掉了。
    • 输入的形状可以是(T×B×* )。T是最长序列长度，B是batch
size，*代表任意维度(可以是0)。如果batch_first=True的话，那么相应的 input size 就是 (B×T×*)。
    • Variable中保存的序列，应该按序列长度的长短排序，长的在前，短的在后。即input[:,0]代表的是最长的序列，input[:,B-1]保存的是最短的序列。
NOTE： 只要是维度大于等于2的input都可以作为这个函数的参数。你可以用它来打包labels，然后用RNN的输出和打包后的labels来计算loss。通过PackedSequence对象的.data属性可以获取 Variable。
参数说明:
    • input (Variable) – 变长序列 被填充后的 batch
    • lengths (list[int]) – Variable 中 每个序列的长度。
    • batch_first (bool, optional) – 如果是True，input的形状应该是B*T*size。
返回值:
    • 一个PackedSequence 对象。
torch.nn.utils.rnn.pad_packed_sequence()
填充packed_sequence。
上面提到的函数的功能是将一个填充后的变长序列压紧。 这个操作和pack_padded_sequence()是相反的。把压紧的序列再填充回来。填充时会初始化为0。
返回的Varaible的值的size是 T×B×*, T 是最长序列的长度，B 是 batch_size,如果 batch_first=True,那么返回值是B×T×*。
Batch中的元素将会以它们长度的逆序排列。
参数说明:
    • sequence (PackedSequence) – 将要被填充的 batch
    • batch_first (bool, optional) – 如果为True，返回的数据的格式为 B×T×*。
返回值: 一个tuple，包含被填充后的序列，和batch中序列的长度列表
一个例子：

输出：（这个输出结果能较为清楚地看到中间过程）

此时PackedSequence对象输入RNN后，输出RNN的还是PackedSequence对象
（最后一个unpacked没有用batch_first, 所以。。。）
