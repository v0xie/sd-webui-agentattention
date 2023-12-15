# sd-webui-agentattention
### An unofficial implementation of Agent Attention in Automatic1111 WebUI.
Speed up image generation with improved image quality using Agent Attention. 

![image](samples/xyz_grid-2419-1-bicycle.png)

### Feature List / Todo
- [x] SD 1.5 Support
- [x] SD XL Support (Requires Max Downsample > 1. If anybody knows the ideal settings to get this to work, please open an Issue!)

### Issues / Pull Requests are welcome!

### Credits
- The authors of the original paper for their method (https://arxiv.org/abs/2312.08874):
	```
	@misc{han2023agent,
      title={Agent Attention: On the Integration of Softmax and Linear Attention}, 
      author={Dongchen Han and Tianzhu Ye and Yizeng Han and Zhuofan Xia and Shiji Song and Gao Huang},
      year={2023},
      eprint={2312.08874},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
	}
	```
- This extension uses code from the official AgentAttention repository: https://github.com/LeapLabTHU/Agent-Attention
- @udon-universe's extension templates (https://github.com/udon-universe/stable-diffusion-webui-extension-templates)

### More samples 
#### SD XL
![image](samples/xyz_grid-2428-1-desk.jpg)
#### SD 1.5
![image](samples/xyz_grid-2418-1-bell%20pepper.png)
![image](samples/xyz_grid-2415-1-desk.png)
![image](samples/xyz_grid-2417-1-goldfinch,%20Carduelis%20carduelis.png)
