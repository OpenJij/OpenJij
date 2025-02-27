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

![](../../../assets/integer_jobs_01.png)

### 問題の一般化

$N$個のタスク$\{0, 1, \dots, N-1\}$と$M$個のコンピュータ$\{0, 1, \dots, M-1\}$を考えましょう。各タスクの実行にかかる時間のリストを$\bm{L} = \{L_0, L_1, \dots, L_{N-1}\}$とします。
$j$番目のコンピュータで実行される仕事の集合を$V_j$としたとき、コンピュータ$j$でタスクを終えるまでの時間は$A_j = \sum_{i \in V_j} L_i$となります。
$i$番目のタスクをコンピュータ$j$で行うことを表すバイナリ変数を$x_{i, j}$とします。

**制約: タスクはどれか1つのコンピュータで実行されなければならない**

例えば、タスク3をコンピュータ1と2の両方で実行することは許されません。これを数式にすると

$$
\sum_{j=0}^{M-1} x_{i, j} = 1 \quad (\forall i \in \{ 0, 1, \dots, N-1 \})
\tag{1}
$$

**目的関数: コンピュータ1の実行時間と他の実行時間の差を小さくする**

コンピュータ1の実行時間を基準とし、それと他のコンピュータの実行時間の差を最小にすることを考えます。これにより実行時間のばらつきが抑えられ、タスクが分散されるようになります。

$$
\min\left\{ \sum_{j=1}^{M-1} (A_1 -A_j)^2\right\} 
\tag{2}
$$

## JijModelingを用いた実装

### 変数の定義

式(1), (2)で用いられている変数を、以下のようにして定義しましょう。


```python
import jijmodeling as jm

# defin variables
L = jm.Placeholder("L", ndim=1)
N = L.len_at(0, latex="N")
M = jm.Placeholder("M")
x = jm.BinaryVar("x", shape=(N, M))
i = jm.Element("i", belong_to=(0, N))
j = jm.Element("j", belong_to=(0, M))
```

`L=jm.Placeholder('L', ndim=1)`でコンピュータに実行させるタスクの実行時間のリストを定義します。
そのリストの長さを`N=L.len_at(0, latex="N")`として定義します。`M`はコンピュータの台数、`x`はバイナリ変数です。
最後に$x_{i, j}$のように、変数の添字として使うものを`i, j`として定義します。

### 制約と目的関数の実装

式(1), (2)を実装しましょう。


```python
# set problem
problem = jm.Problem('Integer Jobs')
# set constraint: job must be executed using a certain node
problem += jm.Constraint('onehot', x[i, :].sum()==1, forall=i)
# set objective function: minimize difference between node 0 and others
problem += jm.sum((j, j!=0), jm.sum(i, L[i]*(x[i, 0]-x[i, j]))**2)
```

`x[i, :].sum()`とすることで、$\sum_j x_{i, j}$を簡潔に実装することができます。

実装した数式をJupyter Notebookで表示してみましょう。


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Integer Jobs} & & \\& & \min \quad \displaystyle \sum_{\substack{j = 0\\j \neq 0}}^{M - 1} \left(\left(\sum_{i = 0}^{N - 1} L_{i} \cdot \left(x_{i, 0} - x_{i, j}\right)\right)^{2}\right) & \\\text{{s.t.}} & & & \\ & \text{onehot} & \displaystyle \sum_{\ast_{1} = 0}^{M - 1} x_{i, \ast_{1}} = 1 & \forall i \in \left\{0,\ldots,N - 1\right\} \\\text{{where}} & & & \\& x & 2\text{-dim binary variable}\\\end{array}$$



### インスタンスの作成

インスタンスを以下のようにします。


```python
# set a list of jobs
inst_L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# set the number of Nodes
inst_M = 3
instance_data = {'L': inst_L, 'M': inst_M}
```

先程の具体例と同様に、$\{1, 2, \dots, 10\}$の10個のタスクを、3台のコンピュータに分散させる状況を考えます。

### JijModeling + OMMXによるQUBOへの変換

ここまで行われてきた実装は、全てJijModelingによるものでした。
これを[PyQUBO](https://pyqubo.readthedocs.io/en/latest/)に変換することで、OpenJijはもちろん、他のソルバーを用いた組合せ最適化計算を行うことが可能になります。


```python
import jijmodeling_transpiler as jmt

# compile
compiled_model = jmt.core.compile_model(problem, instance_data, {})
# get qubo model
pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_model=compiled_model, relax_method=jmt.core.pubo.RelaxationMethod.AugmentedLagrangian)
qubo, const = pubo_builder.get_qubo_dict(multipliers={"onehot": 1.0})
```

### OpenJijによる最適化計算の実行

今回はOpenJijのシミュレーテッド・アニーリングを用いて、最適化問題を解いてみましょう。


```python
import openjij as oj

# set sampler
sampler = oj.SASampler()
# solve problem
response = sampler.sample_qubo(qubo, num_reads=100)
```

`SASampler`を設定し、そのサンプラーに先程作成したQUBOモデルの`qubo`を入力することで、計算結果が得られます。

### デコードと解の表示

計算結果をデコードします。
また実行可能解の中から目的関数値が最小のものを選び出してみましょう。

このようにして得られた結果から、タスク実行が分散されている様子を見てみましょう。


```python
import matplotlib.pyplot as plt
import numpy as np

# decode a result to JijModeling sampleset
sampleset = jmt.core.pubo.decode_from_openjij(response, pubo_builder, compiled_model)
# get feasible samples from sampleset
feasible_samples = sampleset.feasible()
# get the values of objective function of feasible samples
feasible_objectives = [objective for objective in feasible_samples.evaluation.objective]
if len(feasible_objectives) == 0:
    print("No feasible solution found ...")
else:
    # get the index of the loweest objective value
    lowest_index = np.argmin(feasible_objectives)
    # get the indices of x == 1
    tasks, nodes = feasible_samples.record.solution["x"][lowest_index][0]
    # initialize execution time
    exec_time = [0] * inst_M
    # compute summation of execution each nodes
    for i, j in zip(tasks, nodes):
        exec_time[j] += inst_L[i]
    # make plot
    y_axis = range(0, max(exec_time)+1, 2)
    node_names = [str(j) for j in range(inst_M)]
    fig = plt.figure()
    plt.bar(node_names, exec_time)
    plt.yticks(y_axis)
    plt.xlabel('Computer numbers')
    plt.ylabel('Execution time')
    fig.savefig('integer_jobs.png')
```


    
![png](integer_jobs_files/integer_jobs_17_0.png)
    


3つのコンピュータの実行時間がほぼ均等な解が得られました。


