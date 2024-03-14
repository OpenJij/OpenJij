# グラフ彩色問題

こちらでは、[Lucas, 2014, "Ising formulations of many NP problems"](https://doi.org/10.3389/fphy.2014.00005)の 6.1. Graph Coloring を OpenJij と [JijModeling](https://www.ref.documentation.jijzept.com/jijmodeling/)、そして[JijModeling transpiler](https://www.ref.documentation.jijzept.com/jijmodeling-transpiler/) を用いて解く方法をご紹介します。

## 概要: グラフ彩色問題とは

グラフ彩色問題とは、あるグラフ上の辺で繋がれた頂点どうしを異なる色になるように彩色する問題です。NP完全な問題として知られています。

### 具体例

下図のように6個の頂点といくつかの辺からなる無向グラフが与えられたとしましょう。

![](../../../assets/graph_coloring_01.png)

これを3色で全ての頂点を塗り分けると以下のようになります。

![](../../../assets/graph_coloring_02.png)

全ての辺において、その両端に位置する頂点は異なる色で塗り分けられていることがわかります。

### 問題の一般化

それではこの問題を一般化し、数式で表現してみましょう。無向グラフ$G = (V, E)$を、辺で結ばれた頂点の色が重複しないように$N$色で塗り分けることを考えます。
頂点の色分けをバイナリ変数$x_{v, n}$で表すことにします。$v$番目の頂点を$n$の色で塗り分けるとき、$x_{v, n} = 1$、それ以外では$x_{v, n} = 0$となります。  

**制約: 頂点はどれか一色で塗り分けなければならない**

例えば、青色と緑色の2色で1つの頂点を塗ることは許されません。これを数式で表現すると、以下のようになります。


$$
\nonumber
\sum_{n=0}^{N-1} x_{v, n} = 1 \quad (\forall n \in \{ 0, 1, \dots, N-1 \}) \tag{1}
$$

**目的関数: 同じ色の頂点を両端に持つ辺の数を最小にする**

グラフ彩色問題の問題設定から、全ての辺の両端の頂点が異なる色で塗り分けられる必要があります。これを数式で表現すると

$$
\nonumber
\min \quad \sum_{n=0}^{N-1} \sum_{(uv) \in E} x_{u, n} x_{v, n} \tag{2}
$$

もし、全ての辺の両端の頂点が異なる色で塗り分けられているなら、この目的関数値はゼロとなります。

## JijModelingによるモデル構築

### 変数定義

式(1), (2)で用いられている変数を、以下のようにして定義しましょう。


```python
import jijmodeling as jm


# define variables
V = jm.Placeholder('V')
E = jm.Placeholder('E', ndim=2)
N = jm.Placeholder('N')
x = jm.BinaryVar('x', shape=(V, N))
n = jm.Element('i', (0, N))
v = jm.Element('v', (0, V))
e = jm.Element('e', E)
```

`V=jm.Placeholder('V')`でグラフの頂点数、`E=jm.Placeholder('E', ndim=2)`でグラフの辺集合を定義します。
`N=jm.Placeholder('N')`でグラフを塗り分ける色数を定義し、その`V, N`を用いてバイナリ変数$x_{v, n}$を`x=jm.BinaryVar('x', shape=(V, N))`のように定義します。`n, v`はバイナリ変数の添字に用いる変数です。
最後の`e`は辺を表す変数です。`e[0], e[1]`が辺`e`の両端に位置する頂点となります。すなわち$(u, v) = (e[0], e[1])$です。

### 制約の実装

式(1)を実装します。


```python
# set problem
problem = jm.Problem('Graph Coloring')
# set one-hot constraint that each vertex has only one color
problem += jm.Constraint('color', x[v, :].sum()==1, forall=v)
```

問題を作成し、そこに制約を追加しましょう。`x[v, :].sum()`とすることで`Sum(n, x[v, n])`を簡潔に実装することができます。

### 目的関数の追加

式(2)の目的関数を実装しましょう。


```python
# set objective function: minimize edges whose vertices connected by edges are the same color
problem += jm.sum([n, e], x[e[0], n]*x[e[1], n])
```

`jm.sum([n, e], ...)`とすることで、$\sum_n \sum_e$を表現することができます。`x[e[0], n]`は$x_{e[0], n}$、`x[e[1], n]`は$x_{e[1], n}$を表していいます。  
実際に実装された数式をJupyter Notebookで表示してみましょう。


```python
problem
```




$$\begin{array}{cccc}\text{Problem:} & \text{Graph Coloring} & & \\& & \min \quad \displaystyle \sum_{i = 0}^{N - 1} \sum_{e \in E} x_{e_{0}, i} \cdot x_{e_{1}, i} & \\\text{{s.t.}} & & & \\ & \text{color} & \displaystyle \sum_{\ast_{1} = 0}^{N - 1} x_{v, \ast_{1}} = 1 & \forall v \in \left\{0,\ldots,V - 1\right\} \\\text{{where}} & & & \\& x & 2\text{-dim binary variable}\\\end{array}$$



### インスタンスの作成

実際にグラフ彩色を行うグラフを設定しましょう。


```python
import networkx as nx

# set the number of vertices
inst_V = 12
# set the number of colors
inst_N = 4
# create a random graph
inst_G = nx.gnp_random_graph(inst_V, 0.4)
# get information of edges
inst_E = [list(edge) for edge in inst_G.edges]
instance_data = {'V': inst_V, 'N': inst_N, 'E': inst_E, 'G': inst_G}
```

今回は次のようなグラフを塗り分けてみましょう。


```python
import matplotlib.pyplot as plt

nx.draw_networkx(inst_G, with_labels=True)
plt.show()
```


    
![png](graph_coloring_files/graph_coloring_12_0.png)
    


### JijModeling transpilerによるPyQUBOへの変換

ここまで行われてきた実装は、全てJijModelingによるものでした。
これを[PyQUBO](https://pyqubo.readthedocs.io/en/latest/)に変換することで、OpenJijはもちろん、他のソルバーを用いた組合せ最適化計算を行うことが可能になります。


```python
import jijmodeling_transpiler as jmt

# compile
compiled_model = jmt.core.compile_model(problem, instance_data, {})
# get qubo model
pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_model=compiled_model, relax_method=jmt.core.pubo.RelaxationMethod.AugmentedLagrangian)
qubo, const = pubo_builder.get_qubo_dict(multipliers={"color": 1.0})
```

### OpenJijによる最適化計算の実行

今回はOpenJijのシミュレーテッド・アニーリングを用いて、最適化問題を解いてみましょう。


```python
import openjij as oj

# set sampler
sampler = oj.SASampler()
# solve problem
result = sampler.sample_qubo(qubo)
```

`SASampler`を設定し、そのサンプラーに先程作成したQUBOモデルの`qubo`を入力することで、計算結果が得られます。

### デコードと解の表示

計算結果をデコードします。
また実行可能解の中から目的関数値が最小のものを選び出し、それを可視化してみましょう。


```python
import numpy as np

# decode a result to JijModeling sampleset
sampleset = jmt.core.pubo.decode_from_openjij(result, pubo_builder, compiled_model)
# get feasible samples from sampleset
feasible_samples = sampleset.feasible()
# get the values of objective function of feasible samples
feasible_objectives = [objective for objective in feasible_samples.evaluation.objective]
if len(feasible_objectives) == 0:
    print("No feasible solution found ...")
else:
    # get the index of the lowest objective value
    lowest_index = np.argmin(feasible_objectives)
    # get the indices of x == 1
    x_indices = feasible_samples.record.solution["x"][lowest_index][0]
    # initialize a list for color of nodes
    node_colors = [-1] * instance_data["V"]
    # draw the graph
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(instance_data["N"])]
    for pair in zip(*x_indices):
        node_colors[pair[0]] = colors[pair[1]]
    nx.draw_networkx(inst_G, node_color=node_colors, with_labels=True)
    plt.show()
```


    
![png](graph_coloring_files/graph_coloring_18_0.png)
    



