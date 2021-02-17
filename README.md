# ONMF_ONTF_NDL

Online Nonnegative Matrix/Tensor Factorization algorithms with applications in dictionary learning for image and network data. 

For network dictionary learning experiments, we recommend to look at an alternative repository [NDL_paper](https://github.com/HanbaekLyu/NDL_paper/blob/main/README.md) associated with the more recent paper 

Hanbaek Lyu, Yacoub Kureh, Joshua Vendrow, and Mason A. Porter,\
[*"Learning low-rank latent mesoscale structures in networks*"](https://arxiv.org/abs/2102.06984) (arXiv 2021)

## References

These codes are based on my papers below: 
  1. Hanbaek Lyu, Deanna Needell, and Laura Balzano, 
     “Online matrix factorization for markovian data and applications to network dictionary learning.” 
     https://arxiv.org/abs/1911.01931
  2. Hanbaek Lyu, Facundo Memoli, and David Sivakoff, 
     “Sampling random graph homomorphisms and applications to network data analysis.” 
     https://arxiv.org/abs/1910.09483

## File description 

  1. **onmf.py** : Online Nonnegative Matrix Factorization algorithm 
  2. **ontf.py** : Online Nonnegative Tensor Factorization algorithm
  3. **image_reconstruction.py** : Dictionary learning / Image reconstruction based on onmf.py
  4. **image_reconstruction_tensor.py** : Dictionary learning / reconstruction for color images based on ontf.py
  5. **network_reconstruction.py** : Network Dictionary Learning proposed in reference [1] for network data in matrices 
  6. **network_reconstruction_nx.py** : Network Dictionary Learning proposed in reference [1] for networkx format 
  7. **ising_simulator.py** : Gibbs sampler for the 2-dimensional Ising model 
  8. **ising_reconstruction.py** : Dictionary learning / reconstruction from MCMC trajectory of Ising spin configurations 
  
## Authors

* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* My REU students Nick Hanoian and Henry Sojico for polishing up **onmf.py** and **image_reconstruction.py**
