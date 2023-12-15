'''
This code from the following repository: https://github.com/LeapLabTHU/Agent-Attention

@article{han2023agent,
  title={Agent Attention: On the Integration of Softmax and Linear Attention},
  author={Han, Dongchen and Ye, Tianzhu and Han, Yizeng and Xia, Zhuofan and Song, Shiji and Huang, Gao},
  journal={arXiv preprint arXiv:2312.08874},
  year={2023}
}
'''
from . import merge, patch
from .patch import apply_patch, remove_patch

__all__ = ["merge", "patch", "apply_patch", "remove_patch"]