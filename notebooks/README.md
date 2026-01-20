### How to run a notebook on GPU and open in local browser.
1. Make sure you have [inter cluster ssh keys](https://hyak.uw.edu/docs/setup/intracluster-keys/) setup for hyak. This will
let jupyter forward your notebook from a gpu node to the login node. Then you can forward the notebook from the login node
to your local browser.
2. Request a GPU node from klone1 login node with the following command and start a notebook:
```bash
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@klone-login01 ecology-core]$ srun -A stf -p ckpt --nodes=1 --ntasks-per-node=1 --mem=48G --time=8:00:00 --cpus-per-task=1 --gpus-per-node=1 --pty /bin/bash
srun: job 16052645 queued and waiting for resources
srun: job 16052645 has been allocated resources
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@${GPU_NODE_ID} ecology-core]$ jupyter notebook --port=8066 --ip=0.0.0.0
[I 14:32:02.806 NotebookApp] Serving notebooks from local directory: ecology-core
[I 14:32:02.806 NotebookApp] Jupyter Notebook 6.4.10 is running at:
[I 14:32:02.806 NotebookApp] http://g3016:8066/?token=${NOTEBOOK_TOKEN}
[I 14:32:02.806 NotebookApp]  or http://127.0.0.1:8066/?token=${NOTEBOOK_TOKEN}
```
3. In another terminal on your personal computer, forward the port using this [instruction](https://hyak.uw.edu/docs/setup/portforwarding#openssh-client-linuxmacconemu). Replace the `GPU_NODE_ID` with that for the assigned GPU.

```bash
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@klone-login01 ecology-core]$ ssh -L 8066:${GPU_NODE_ID}:8066 klone1.hyak.uw.edu           Password:
Duo two-factor login for emazuh
Enter a passcode or select one of the following options:
 1. Duo Push to XXX-XXX-9732
 2. Phone call to XXX-XXX-9732
Passcode or option (1-2): 1
```
If you receive an address already in use error, you can kill the offending process as below and try forwarding again:
```bash
bind [127.0.0.1]:8066: Address already in use
channel_setup_fwd_listener_tcpip: cannot listen to port: 8066
Could not request local forwarding.
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@klone-login01 ecology-core]$ netstat -anpt | grep '8066'
(Not all processes could be identified, non-owned process info
 will not be shown, you would have to be root to see it all.)
tcp        0      0 127.0.0.1:8066          0.0.0.0:*               LISTEN      1168057/ssh
tcp6       0      0 ::1:8066                :::*                    LISTEN      1168057/ssh
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@klone-login01 ecology-core]$ kill -9 1168057
(/gscratch/scrubbed/emazuh/ecologyenv) [emazuh@klone-login01 ecology-core]$ ssh -NfL 8066:${GPU_NODE_ID}.hyak.local:8066 klone1.hyak.uw.edu
```
4. You should now be able to open the notebook through Visual Studio Code (either with the open in browser pop up or by using `CTRL-CMD-P`, select `Forward a port` and type `8066`. If prompted, use the `NOTEBOOK_TOKEN` to gain access.
