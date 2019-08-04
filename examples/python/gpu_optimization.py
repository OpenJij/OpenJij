import cxxjij


def gen_chimera_testcase(J):
    import cxxjij.graph as G
    J[0,0,0,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,0,0,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,0,0,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,0,0,G.ChimeraDir.IN_3or7] = +0.25;
    J[0,0,1,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,0,1,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,0,1,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,0,1,G.ChimeraDir.IN_3or7] = +0.25;
    J[0,0,2,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,0,2,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,0,2,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,0,2,G.ChimeraDir.IN_3or7] = +0.25;
    J[0,0,3,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,0,3,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,0,3,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,0,3,G.ChimeraDir.IN_3or7] = +0.25;

    J[0,1,0,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,1,0,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,1,0,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,1,0,G.ChimeraDir.IN_3or7] = +0.25;
    J[0,1,1,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,1,1,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,1,1,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,1,1,G.ChimeraDir.IN_3or7] = +0.25;
    J[0,1,2,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,1,2,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,1,2,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,1,2,G.ChimeraDir.IN_3or7] = +0.25;
    J[0,1,3,G.ChimeraDir.IN_0or4] = +0.25;
    J[0,1,3,G.ChimeraDir.IN_1or5] = +0.25;
    J[0,1,3,G.ChimeraDir.IN_2or6] = +0.25;
    J[0,1,3,G.ChimeraDir.IN_3or7] = +0.25;

    J[1,0,0,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,0,0,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,0,0,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,0,0,G.ChimeraDir.IN_3or7] = +0.25;
    J[1,0,1,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,0,1,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,0,1,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,0,1,G.ChimeraDir.IN_3or7] = +0.25;
    J[1,0,2,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,0,2,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,0,2,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,0,2,G.ChimeraDir.IN_3or7] = +0.25;
    J[1,0,3,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,0,3,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,0,3,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,0,3,G.ChimeraDir.IN_3or7] = +0.25;

    J[1,1,0,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,1,0,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,1,0,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,1,0,G.ChimeraDir.IN_3or7] = +0.25;
    J[1,1,1,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,1,1,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,1,1,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,1,1,G.ChimeraDir.IN_3or7] = +0.25;
    J[1,1,2,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,1,2,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,1,2,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,1,2,G.ChimeraDir.IN_3or7] = +0.25;
    J[1,1,3,G.ChimeraDir.IN_0or4] = +0.25;
    J[1,1,3,G.ChimeraDir.IN_1or5] = +0.25;
    J[1,1,3,G.ChimeraDir.IN_2or6] = +0.25;
    J[1,1,3,G.ChimeraDir.IN_3or7] = +0.25;

    J[0,0,0] = +1;

    J[0,0,6,G.ChimeraDir.PLUS_C] = +1;
    J[0,0,3,G.ChimeraDir.PLUS_R] = -1;
    J[1,0,5,G.ChimeraDir.PLUS_C] = +1;

    true_chimera_spin = [0] * J.size()
    
    true_chimera_spin[J.to_ind(0,0,0)] = -1;
    true_chimera_spin[J.to_ind(0,0,1)] = -1;
    true_chimera_spin[J.to_ind(0,0,2)] = -1;
    true_chimera_spin[J.to_ind(0,0,3)] = -1;
    true_chimera_spin[J.to_ind(0,0,4)] = +1;
    true_chimera_spin[J.to_ind(0,0,5)] = +1;
    true_chimera_spin[J.to_ind(0,0,6)] = +1;
    true_chimera_spin[J.to_ind(0,0,7)] = +1;

    true_chimera_spin[J.to_ind(0,1,0)] = +1;
    true_chimera_spin[J.to_ind(0,1,1)] = +1;
    true_chimera_spin[J.to_ind(0,1,2)] = +1;
    true_chimera_spin[J.to_ind(0,1,3)] = +1;
    true_chimera_spin[J.to_ind(0,1,4)] = -1;
    true_chimera_spin[J.to_ind(0,1,5)] = -1;
    true_chimera_spin[J.to_ind(0,1,6)] = -1;
    true_chimera_spin[J.to_ind(0,1,7)] = -1;

    true_chimera_spin[J.to_ind(1,0,0)] = -1;
    true_chimera_spin[J.to_ind(1,0,1)] = -1;
    true_chimera_spin[J.to_ind(1,0,2)] = -1;
    true_chimera_spin[J.to_ind(1,0,3)] = -1;
    true_chimera_spin[J.to_ind(1,0,4)] = +1;
    true_chimera_spin[J.to_ind(1,0,5)] = +1;
    true_chimera_spin[J.to_ind(1,0,6)] = +1;
    true_chimera_spin[J.to_ind(1,0,7)] = +1;

    true_chimera_spin[J.to_ind(1,1,0)] = +1;
    true_chimera_spin[J.to_ind(1,1,1)] = +1;
    true_chimera_spin[J.to_ind(1,1,2)] = +1;
    true_chimera_spin[J.to_ind(1,1,3)] = +1;
    true_chimera_spin[J.to_ind(1,1,4)] = -1;
    true_chimera_spin[J.to_ind(1,1,5)] = -1;
    true_chimera_spin[J.to_ind(1,1,6)] = -1;
    true_chimera_spin[J.to_ind(1,1,7)] = -1;

    return J, true_chimera_spin

    

def main():
    chimera = cxxjij.graph.ChimeraGPU(2, 2)
    chimera, chimera_ground = gen_chimera_testcase(chimera)
    # system = cxxjij.system.make_chimera_transverse_gpu(
    #     chimera.gen_spin(), chimera, 1.0, 10
    # )
    system = cxxjij.system.make_chimera_transverse_gpu(
        [chimera.gen_spin() for _ in range(4)],
        chimera, gamma=1.0
    )

if __name__ == "__main__":
    main()
