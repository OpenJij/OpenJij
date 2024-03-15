# ナップサック問題

こちらでは、[Lucas, 2014, "Ising formulations of many NP problems"](https://doi.org/10.3389/fphy.2014.00005)の 5.2. Knapsack with Integer Weights を OpenJij と [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/)、そして[JijModeling transpiler](https://www.ref.documentation.jijzept.com/jijmodeling-transpiler/) を用いて解く方法をご紹介します。

## 概要: ナップサック問題とは

ナップサック問題は、具体的には以下のような状況で最適解を求める問題です。
最も有名なNP困難な整数計画問題の一つとして知られています。まずは具体例を考えてみましょう。

### 具体例

この問題の具体例として、以下のような物語を考えます。

> ある探検家がある洞窟を探検していました。しばらく洞窟の中を歩いていると、思いがけなく複数の宝物を発見しました。

||宝物A|宝物B|宝物C|宝物D|宝物E|宝物F|
|---|---|---|---|---|---|---|
|値段|$5000|$7000|$2000|$1000|$4000|$3000|
|重さ|800g|1000g|600g|400g|500g|300g|

> しかし探検家の手持ちの荷物の中で宝物を運べるような袋としては、残念ながら小さなナップサックしか持ち合わせていませんでした。
> このナップサックには2kgの荷物しか入れることができません。探検家はこのナップサックに入れる宝物の価値をできるだけ高くしたいのですが、どの荷物を選べば最も効率的に宝物を持って帰ることができるでしょうか。

### 問題の一般化

この問題を一般化するには、ナップサックに入れる荷物$N$個の集合$\{ 0, 1, \dots, i, \dots, N-1\}$があり、各荷物が$i$をインデックスとして持っているものとして考えます。  
ナップサックに入れる各荷物$i$のコストのリスト$\bm{v}$と重さのリスト$\bm{w}$を作ることで、問題を表現することができます。

$$
\nonumber
\bm{v} = \{v_0, v_1, \dots, v_i, \dots, v_{N-1}\}
$$

$$
\nonumber
\bm{w} = \{w_0, w_1, \dots, w_i, \dots, w_{N-1}\}
$$

さらに$i$番目の荷物を選んだことを表すバイナリ変数を$x_i$としましょう。この変数は$i$をナップサックに入れるとき$x_i = 1$、入れないとき$x_i = 0$となるような変数です。最後にナップサックの最大容量を$W$とします。  
最大化したいのは、ナップサックに入れる荷物の価値の合計です。よってこれを目的関数として表現しましょう。さらにナップサックの容量制限以下にしなければならない制約を考えると、ナップサック問題は以下のような数式で表現されます。

$$
\max \quad \sum_{i=0}^{N-1} v_i x_i \tag{1}
$$

$$
\mathrm{s.t.} \quad \sum_{i=0}^{N-1} w_i x_i \leq W \tag{2}
$$

$$
x_i \in \{0, 1\} \quad (\forall i \in \{0, 1, \dots, N-1\}) \tag{3}
$$

## JijModelingによるモデル構築

### 変数定義

ナップサック問題で用いられている変数$\bm{v}, \bm{w}, N, W, x_i, i$を、以下のようにして定義しましょう。


```python
import jijmodeling as jm

# define variables
v = jm.Placeholder('v', ndim=1)
w = jm.Placeholder('w', ndim=1)
W = jm.Placeholder('W')
N = v.len_at(0, latex="N")
x = jm.BinaryVar('x', shape=(N,))
i = jm.Element('i', (0, N))
```

`v = jm.Placeholder('v', ndim=1), w = jm.Placeholder('w', ndim=1)`でナップサックに入れる物の価値と重さを表現する一次元のリストを定義し、さらに`W = jm.Placeholder('W')`ではナップサックの容量制限を表す$W$を定義しています。
`N=v.len_at(0, latex="N")`のようにすることで、先ほど定義した`v`の要素数を取得しています。
これを用いて、最適化に用いるバイナリ変数のリストを`x = jm.Binary('x', shape=(N))`を定義します。
最後の`i = jm.Element('i', (0, N))`により$0 \leq i < N$を満たす$v_i, w_i, x_i$の添字を定義します。

### 制約と目的関数の実装

式(1), (2)を実装しましょう。


```python
# set problem
problem = jm.Problem('Knapsack', sense=jm.ProblemSense.MAXIMIZE) 
# set constraint: less than the capacity
problem += jm.Constraint("capacity", jm.sum(i, w[i]*x[i]) <= W)
# set objective function
problem += jm.sum(i, v[i]*x[i])
```

`jm.sum(i, ...)`のようにすることで、$\sum_i$を実装することができます。
`jm.Constraint("capacity", ...)`のようにすることで、"capacity"という名の制約を設定しています。　　
実装した数理モデルを、Jupyter Notebookで表示してみましょう。

`Constraint(制約名, 制約式)`とすることで、制約式に適当な制約名を付与することができます。  
実際に実装された数式をJupyter Notebookで表示してみましょう。


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Knapsack} & & \\& & \max \quad \displaystyle \sum_{i = 0}^{N - 1} v_{i} \cdot x_{i} & \\\text{{s.t.}} & & & \\ & \text{capacity} & \displaystyle \sum_{i = 0}^{N - 1} w_{i} \cdot x_{i} \leq W &  \\\text{{where}} & & & \\& x & 1\text{-dim binary variable}\\\end{array}$$



### インスタンスの作成

先程の冒険家の物語を、インスタンスとして設定しましょう。
ただし物の価値は$1000で規格化、さらに物の重さも100gで規格化された値を用います。


```python
import numpy as np

# set a list of values & weights 
inst_v = np.array([5, 7, 2, 1, 4, 3])
inst_w = np.array([8, 10, 6, 4, 5, 3])
# set maximum weight
inst_W = 20
instance_data = {'v': inst_v, 'w': inst_w, 'W': inst_W} 
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
qubo, const = pubo_builder.get_qubo_dict(multipliers={"capacity": 1.0})
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
また実行可能解の名から目的関数値が最大のものを選び出してみましょう。


```python
# decode a result to JijModeling sampleset
sampleset = jmt.core.pubo.decode_from_openjij(response, pubo_builder, compiled_model)
# get feasible samples from sampleset
feasible_samples = sampleset.feasible()
# get the values of objective function of feasible samples
feasible_objectives = [objective for objective in feasible_samples.evaluation.objective]
if len(feasible_objectives) == 0:
    print("No feasible solution found ...")
else:
    # get the index of the highest objective value
    highest_index = np.argmax(feasible_objectives)
    # get the indices of x == 1
    x_indices = np.array(feasible_samples.record.solution["x"][highest_index][0][0])
    print("Indices of x == 1: ", x_indices)
    print('Value of objective function: ', feasible_objectives[highest_index])
    print('Value of constraint term: ', feasible_samples.evaluation.constraint_violations["capacity"][highest_index])
    print('Total weight: ', np.sum(inst_w[x_indices]))
```

    Indices of x == 1:  [1 4 5]
    Value of objective function:  14.0
    Value of constraint term:  0.0
    Total weight:  18

