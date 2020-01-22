# Project Title

Online Nonnegative Matrix/Tensor Factorization algorithms with applications in dictionary learning for image and network data. 

## References

These codes are based on my papers below: 
  1. Hanbaek Lyu, Deanna Needell, and Laura Balzano, 
     “Online matrix factorization for markovian data and applications to network dictionary learning.” 
     https://arxiv.org/abs/1911.01931
  2. Hanbaek Lyu, Facundo Memoli, and David Sivakoff, 
     “Sampling random graph homomorphisms and applications to network data analysis.” 
     https://arxiv.org/abs/1910.09483
  3. Hanbaek Lyu, Deanna Needell, and Chris Strohmeier, 
     “Online tensor factorization for markovian data”
     In preperation.

## File description 

  1. **onmf.py** : Online Nonnegative Matrix Factorization algorithm 
  2. **ontf.py** : Online Nonnegative Tensor Factorization algorithm
  3. **image_reconstruction.py** : Dictionary learning / Image reconstruction based on onmf.py
  4. **image_reconstruction_tensor.py** : Dictionary learning / reconstruction for color images based on ontf.py
  5. **network_reconstruction.py** : Network Dictionary Learning proposed in reference [1] for network data in matrices 
  6. **network_reconstruction_nx.py** : Network Dictionary Learning proposed in reference [1] for networkx format 
  
## Authors

* **Hanbaek Lyu** - *Initial work* - [Website](https://hanbaeklyu.com)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* My REU students Nick Hanoian and Henry Sojico for polishing up **onmf.py** and **image_reconstruction.py**
