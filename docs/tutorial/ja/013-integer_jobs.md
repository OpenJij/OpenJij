# 整数長ジョブシーケンス問題

こちらでは、[Lucas, 2014, "Ising formulations of many NP problems"](https://doi.org/10.3389/fphy.2014.00005)の 6.3. Job Sequencing with Integer Lengths を OpenJij と [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/)、そして[JijModeling transpiler](https://www.ref.documentation.jijzept.com/jijmodeling-transpiler/) を用いて解く方法をご紹介します。

## 概要: 整数長ジョブシーケンス問題とは

タスク1は実行するのに1時間、タスク2は実行に3時間、というように、整数の長さを持つタスクがいくつかあったとします。
これらを複数の実行するコンピュータに配分するとき、偏りを作ることなくコンピュータの実行時間を分散させるにはどのような組合せがあるでしょうか、というのを考える問題です。

### 具体例

分かりやすくするために具体的に以下のような状況を考えてみましょう。 

> ここに10個のタスクと3個のコンピュータがあります。10個の仕事の長さはそれぞれ$1, 2, \dots, 10$とします。
> これらのタスクをどのようにコンピュータに仕事を割り振れば仕事にかかる時間の最大値を最小化できるか考えます。
> この場合、例えば1つ目のコンピュータには9, 10、2つ目には1, 2, 7, 8、3つ目には3, 4, 5, 6とするととなり、3つのコンピュータの実行時間の最大値は19となり、これが最適解です。

![](../../assets/integer_jobs_01.png)

### 問題の一般化

$N$個のタスク$\{0, 1, \dots, N-1\}$と$M$個のコンピュータ$\{0, 1, \dots, M-1\}$を考えましょう。各タスクの実行にかかる時間のリストを$\bm{L} = \{L_0, L_1, \dots, L_{N-1}\}$とします。
$j$番目のコンピュータで実行される仕事の集合を$V_j$としたとき、コンピュータ$j$でタスクを終えるまでの時間は$A_j = \sum_{i \in V_j} L_i$となります。
$i$番目のタスクをコンピュータ$j$で行うことを表すバイナリ変数を$x_{i, j}$とします。

**制約: タスクはどれか1つのコンピュータで実行されなければならない**

例えば、タスク3をコンピュータ1と2の両方で実行することは許されません。これを数式にすると

$$
\nonumber
\sum_{j=0}^{M-1} x_{i, j} = 1 \quad (\forall i \in \{ 0, 1, \dots, N-1 \})
$$ (1)

**目的関数: コンピュータ1の実行時間と他の実行時間の差を小さくする**

コンピュータ1の実行時間を基準とし、それと他のコンピュータの実行時間の差を最小にすることを考えます。これにより実行時間のばらつきが抑えられ、タスクが分散されるようになります。

$$
\nonumber
\min\left\{ \sum_{j=1}^{M-1} (A_1 -A_j)^2\right\} 
$$ (2)

## JijModelingによるモデル構築

### 整数長ジョブシーケンス問題で用いる変数を定義

式(1), (2)で用いられている変数を、以下のようにして定義しましょう。

```python
import jijmodeling as jm


# defin variables
L = jm.Placeholder('L', dim=1)
N = L.shape[0]
M = jm.Placeholder('M')
x = jm.Binary('x', shape=(N, M))
i = jm.Element('i', (0, N))
j = jm.Element('j', (0, M))
```

`L=jm.Placeholder('L', dim=1)`でコンピュータに実行させるタスクの実行時間のリストを定義します。
そのリストの長さを`N=L.shape[0]`として定義します。`M`はコンピュータの台数、`x`はバイナリ変数です。最後に$x_{i, j}$のように、変数の添字として使うものを`i, j`として定義します。

### 制約の追加

式(1)を制約として実装します。

```python
# set problem
problem = jm.Problem('Integer Jobs')
# set constraint: job must be executed using a certain node
onehot = x[i, :]
problem += jm.Constraint('onehot', onehot==1, forall=i)
```

問題を作成し、そこに制約を追加しましょう。`x[i, :]`とすることで`Sum(j, x[i, j])`を簡潔に実装することができます。

### 目的関数の追加

式(2)の目的関数を実装しましょう。

```python
# set objective function: minimize difference between node 0 and others
diffj = jm.Sum(i, L[i]*x[i, 0]) - jm.Sum(i, L[i]*x[i, j])
sumdiff2 = jm.Sum((j, j!=0), diffj*diffj)
problem += sumdiff2
```

`diffj`で$A_1 - A_j$を計算し、それを2乗して総和を取ったものを制約とします。  
実際に実装された数式をJupyter Notebookで表示してみましょう。

![](../../assets/integer_jobs_02.png)

### インスタンスの作成

実際に実行するタスクなどを設定しましょう。

```python
# set a list of jobs
inst_L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# set the number of Nodes
inst_M = 3
instance_data = {'L': inst_L, 'M': inst_M}
```

先程の具体例と同様に、$\{1, 2, \dots, 10\}$の10個のタスクを3台のコンピュータで実行することを考えます。

### 未定乗数の設定

整数長ジョブシーケンスには制約が一つあります。よってその制約の重みを設定する必要があります。
先程の`Constraint`部分で付けた名前と一致させるように、辞書型を用いて設定を行います。

```python
# set multipliers
lam1 = 1.0
multipliers = {'onehot': lam1}    
```

### JijModeling transpilerによるPyQUBOへの変換

ここまで行われてきた実装は、全てJijModelingによるものでした。
これを[PyQUBO](https://pyqubo.readthedocs.io/en/latest/)に変換することで、OpenJijはもちろん、他のソルバーを用いた組合せ最適化計算を行うことが可能になります。

```python
from jijmodeling.transpiler.pyqubo import to_pyqubo

# convert to pyqubo
pyq_model, pyq_chache = to_pyqubo(problem, instance_data, {})
qubo, bias = pyq_model.compile().to_qubo(feed_dict=multipliers)
```

JijModelingで作成された`problem`、そして先ほど値を設定した`instance_data`を引数として、`to_pyqubo`によりPyQUBOモデルを作成します。次にそれをコンパイルすることで、OpenJijなどで計算が可能なQUBOモデルにします。

### OpenJijによる最適化計算の実行

今回はOpenJijのシミュレーテッド・アニーリングを用いて、最適化問題を解くことにします。
それには以下のようにします。

```python
import openjij as oj

# set sampler
sampler = oj.SASampler()
# solve problem
response = sampler.sample_qubo(qubo)
```    

`SASampler`を設定し、そのサンプラーに先程作成したQUBOモデルの`qubo`を入力することで、計算結果が得られます。

### デコードと解の表示

返された計算結果をデコードし、解析を行いやすくします。

```python
# decode solution
result = pyq_chache.decode(response)
```

このようにして得られた結果から、タスク実行が分散されている様子を見てみましょう。

```python
# extract feasible solution
feasible = result.feasible()
# get the index of the lowest objective function
objectives = feasible.evaluation.objective
obs_dict = {i: objectives[i] for i in range(len(objectives))}
lowest = min(obs_dict, key=obs_dict.get)
# get indices of x = 1
indices, _, _ = feasible.record.solution['x'][lowest]
# get task number and execution node
tasks, nodes = indices
# get instance information
L = instance_data['L']
M = instance_data['M']
# initialize execution time
exec_time = [0] * M
# compute summation of execution time each nodes
for i, j in zip(tasks, nodes):
    exec_time[j] += L[i]
y_axis = range(0, max(exec_time)+1, 2)
node_names = [str(j) for j in range(M)]
fig = plt.figure()
plt.bar(node_names, exec_time)
plt.yticks(y_axis)
plt.xlabel('Computer numbers')
plt.ylabel('Execution time')
fig.savefig('integer_jobs.png')
```

すると以下のように、3つのコンピュータの実行時間がほぼ均等な解が得られます。

![](../../assets/integer_jobs_03.png)
