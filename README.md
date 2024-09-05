# llama-2-Finetuning-Alteon

---
 
### Features
- Instruction finetune using HuggingFace transformer library
- 4 bit quantization with bitandbytes
- Training LORA (QLORA) adapters with PEFT and SFT

---

### Huggingface Hub

Finetuned model available here : https://huggingface.co/bridge4/Llama-2-7b-QLoRA-Alteoncmds

Training Dataset : https://huggingface.co/datasets/bridge4/Alteoncmdsdataset

---

### Training Dataset

![image](https://github.com/user-attachments/assets/dfec7eb4-2877-4dfe-beb3-f09857856e3f)


---

### Files

**Notebook to create dataset for instruction finetuning llama2 chat model using approprite formatting and prompts. Use the notebook to create your own custom dataset for finetuning** 

https://github.com/perrin-ay/llama-2-7b-Finetuning-Alteon/blob/main/Alteon_commands_dataset_for_Instruction_fine_tuning.ipynb
 
**Scripts for finetuning**

https://github.com/perrin-ay/llama-2-7b-Finetuning-Alteon/blob/main/llama_2_7b_sft_qlora_on_alteon_cli_commands.py

---

### Inference

```
<s>[INST] ### Instruction: what are the commands to configure a virt called vgdrf with ip address 10.107.246.1 and service ftp [/INST] ### Response: /cfg/slb/virt vgdrf/ena/vip 10.107.246.1/service ftp

<s>[INST] ### Instruction: commands for virtual server named 80 with vip address 10.105.246.110 and service https and group id hy77 [/INST] ### Response: /cfg/slb/virt 80/ena/vip 10.105.246.110/service 443/group hy77

<s>[INST] ### Instruction: what are the commands to configure a real server called 34 and real ip 1.1.1.1 [/INST] ### Response: /cfg/real 34/ena/rip 1.1.1.1

<s>[INST] ### Instruction: configure real id 34 and ip 192.168.43.4 [/INST] ### Response: /cfg/real 34/ip 192.168.43.4

<s>[INST] ### Instruction: configure real server called 34 and ip 8.89.98.2 [/INST] ### Response: /cfg/real 34/ip 89.98.98.2

<s>[INST] ### Instruction:  commands for alteon virt with id 69769, vip 192.1.2.4, service https and group called ki [/INST] ### Response: /cfg/slb/virt 69769/ena/vip 192.1.2.4/service https/group ki

<s>[INST] ### Instruction: what are the commands to configure a virtual server called vamp and with vip 10.5.3.2 and service https and group called bast [/INST] ### Response: /cfg/vamp/service https/group bast

<s>[INST] ### Instruction: can you get me commands for an alteon group called named 90_can that with real servers test1 and test2 [/INST] ### Response: /cfg/group 90_can/add test1/add test2

<s>[INST] ### Instruction: what are the commands to configure a virtual server with ip address 10.5.3.2 called 20 with service https and group called bast [/INST] ### Response: /cfg/slb/virt 20/ena/vip 10.5.3.2/service 443/group bast
```
