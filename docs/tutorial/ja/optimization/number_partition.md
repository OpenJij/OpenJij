# 数分割問題

こちらでは、[Lucas, 2014, "Ising formulations of many NP problems"](https://doi.org/10.3389/fphy.2014.00005)の 2.1. Number Partitioning を OpenJij と [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/)、そして[JijModeling transpiler](https://www.ref.documentation.jijzept.com/jijmodeling-transpiler/) を用いて解く方法をご紹介します。

## 概要: 数分割問題とは

数分割問題は、与えられた数字の集合を足した合計値が等しくなるように2つの集合に分割する問題です。
ここで、簡単な例を考えてみましょう。

例えば、$A=\{1,2,3,4\}$という数字の集合$A$があるとします。
この集合を合計値が等しくなるように分割するのは簡単で、$\{1,4\},\{2,3\}$とすれば、それぞれの集合の合計値が5になるということがわかります。
このように、集合のサイズが小さい場合には、比較的簡単に答えがもとまりますが、これが大きくなるとすぐには解けません。
そこで、このチュートリアルでは、この問題をアニーリングを使って解いてみましょう。  
まず初めに、この問題のハミルトニアンを考えます。
分割する集合を$A$とし、その要素を$a_i (i = \{0,1,\dots,N-1\})$とします。
ここで$N$はこの集合の要素数です。
この集合$A$を二つの集合を$A_0$と$A_1$に分割するとします。
この時、$x_i$を$A$の$i$番目の要素が、集合$A_0$に含まれる時0、$A_1$に含まれる時1となる変数とします。
この変数$x_i$を用いると、$A_0$に入っている数の合計値は$\sum_i a_i (1-x_i)$とかけ、$A_1$の$\sum_i a_i x_i$となることがわかります。
この問題は、$A_0$と$A_1$に含まれている数の合計値が等しくなるという制約を満たす解を求める問題ですので、これを式にすると、

$$\sum_i a_i (1-x_i)=\sum_i a_i x_i$$

という制約条件を満たす$x_i$を求めよという問題であることがわかります。
これを式変形すると、$\sum_i a_i (1-2x_i)=0$と書くことができ、さらに、Penalty法を用いて、この制約条件を2乗したものをハミルトニアンとすると、結局、数分割問題のハミルトニアンは、

$$
H=\left( \sum_{i=0}^{N-1} a_i (1-2x_i)\right)^2 \tag{1}
$$

となります。

## JijModelingによるモデル構築

式(1)をJijModelingを用いて定式化していきます。


```python
import jijmodeling as jm

problem = jm.Problem("Number Partition")
a = jm.Placeholder("a", ndim=1)
N = a.len_at(0, latex="N")
x = jm.BinaryVar("x", shape=(N, ))
i = jm.Element("i", (0, N))
problem += jm.sum(i, a[i]*(1-2*x[i])) ** 2
```

Jupyter Notebookで実装の確認を行いましょう。


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Number Partition} & & \\& & \min \quad \displaystyle \left(\left(\sum_{i = 0}^{N - 1} a_{i} \cdot \left(- 2 \cdot x_{i} + 1\right)\right)^{2}\right) & \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



### インスタンスデータの作成
ここでは、1から40までの数字を分割する問題を考えましょう。
$N_{i}$から$N_{f}$まで連続する数を分割する問題(連続する数の合計数が偶数の時)では、分割の仕方はいろんなパターンがありますが分割された集合の合計値は

$$\mathrm{total\ value} = \frac{(N_{i} + N_{f})(N_{f} - N_{i} + 1)}{4}$$

と計算することができます。
今の場合には、合計値は410となります。
実際にそれを確かめてみましょう。


```python
import numpy as np

inst_N = 40
instance_data = {"a": np.arange(1, inst_N+1)}
```

### JijModeling transpilerによるPyQUBOへの変換

ここまで行われてきた実装は、全てJijModelingによるものでした。
これを[PyQUBO](https://pyqubo.readthedocs.io/en/latest/)に変換することで、OpenJijはもちろん、他のソルバーを用いた組合せ最適化計算を行うことが可能になります。


```python
import jijmodeling_transpiler as jmt

# compile
compiled_model = jmt.core.compile_model(problem, instance_data, {})
# get qubo model
pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_model=compiled_model, relax_method=jmt.core.pubo.RelaxationMethod.AugmentedLagrangian)
qubo, const = pubo_builder.get_qubo_dict(multipliers={})
```

### OpenJijによる最適化計算の実行
OpenJijを用いて計算してみます。


```python
import openjij as oj

sampler = oj.SASampler()
response = sampler.sample_qubo(qubo, num_reads=1)
```

### デコードと解の表示

計算結果をデコードします。
ここでは、$A$の中で$A_1$に分類されたindexと$A_0$に分類されたindexを分けて、それらについて和をとっています。


```python
# decode a result to JijModeling sampleset
sampleset = jmt.core.pubo.decode_from_openjij(response, pubo_builder, compiled_model)
# get the indices of x == 1
class_1_indices = sampleset.record.solution["x"][0][0][0]
class_0_indices = [i for i in range(0, inst_N) if i not in class_1_indices]

class_1 = instance_data['a'][class_1_indices]
class_0 = instance_data['a'][class_0_indices]

print(f"class 1 : {class_1} , total value = {np.sum(class_1)}")
print(f"class 0 : {class_0} , total value = {np.sum(class_0)}")
```

    class 1 : [ 2  8  9 12 15 18 19 21 22 23 24 26 30 31 35 36 39 40] , total value = 410
    class 0 : [ 1  3  4  5  6  7 10 11 13 14 16 17 20 25 27 28 29 32 33 34 37 38] , total value = 410


我々の予想通り、それぞれの合計値410が得られていることがわかりました。
