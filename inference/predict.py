"""Inference only: load a packaged checkpoint and an actual contract JSON."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
from architecture import InferenceModel
from features import static_features, event_features
from ewc_head import EWCConfig, EWCHead

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def contract_sample(contract,ratio):
    if ratio not in (5,20,50):raise ValueError('ratio must be 5, 20, or 50')
    all_events=contract['erc20_transfer_events']
    if len(all_events)!=contract['event_count']:raise ValueError('Event count mismatch')
    events=all_events[:max(1,math.ceil(len(all_events)*ratio/100))]
    deploy=int(contract['deployment_block'])
    query=int(events[-1]['block_number']) if events else deploy
    return dict(key=contract['contract_id'],bytecode=contract['runtime_bytecode'],
                events=events,deploy=deploy,query=query,ratio=ratio)

def raw(sample):
    return static_features(sample['bytecode']),event_features(sample['events'],sample['deploy'],sample['query'],sample.get('deployer',''))

class Predictor:
    def __init__(self,checkpoint,device='cpu'):
        checkpoint=Path(checkpoint).resolve();card=json.loads((checkpoint.parent/'MODEL_CARD.json').read_text())
        if sha(checkpoint)!=card['checkpoint_sha256']:raise ValueError('Checkpoint checksum mismatch')
        obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
        self.device=device;self.card=card;self.head=None
        if card['kind']=='ewc_head':
            root=checkpoint.parents[3]
            self.encoder=Predictor(root/card['encoder_checkpoint'],device)
            config=EWCConfig(**card['head_config'])
            self.head=EWCHead(config).to(device).eval()
            self.head.load_state_dict(obj['state_dict'],strict=True)
            self.model=self.encoder.model;self.normalization=self.encoder.normalization
        else:
            c=dict(obj['config']);c.setdefault('aggregation','weighted')
            state={k:v for k,v in obj['state_dict'].items() if not k.startswith(('predict_sd.','predict_ds.'))}
            self.model=InferenceModel(c,state).to(device).eval()
            self.normalization={k:v.cpu().numpy() if torch.is_tensor(v) else np.asarray(v) for k,v in obj['normalization'].items()}
        for p in self.model.parameters():
            if not torch.isfinite(p).all():raise ValueError('Nonfinite model weight')
        for name,n in [('mean_static',264),('std_static',264),('mean_event',16),('std_event',16)]:
            v=self.normalization[name]
            if v.shape!=(n,) or not np.isfinite(v).all() or (name.startswith('std_') and (v<=0).any()):raise ValueError('Invalid normalization')

    def batch(self,samples):
        pairs=[raw(s) for s in samples];n=self.normalization
        static=np.stack([(s-n['mean_static'])/n['std_static'] for s,e in pairs]).astype(np.float32)
        length=max(1,max(len(e) for s,e in pairs));events=np.zeros((len(samples),length,16),np.float32);mask=np.zeros((len(samples),length),bool)
        for i,(s,e) in enumerate(pairs):
            if len(e):events[i,:len(e)]=(e-n['mean_event'])/n['std_event'];mask[i,:len(e)]=True
        return tuple(torch.as_tensor(v,device=self.device) for v in (static,events,mask))

    @torch.inference_mode()
    def predict(self,samples):
        if self.head is None:out=self.model(*self.batch(samples))
        else:
            captured=[]
            hook=self.model.head.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
            try:self.model(*self.batch(samples))
            finally:hook.remove()
            d=self.head.details(captured[0]);score=d['distances'][:,0]-d['distances'][:,1]
            out=dict(score=score,probability=d['logits'].softmax(-1)[:,1])
        return {k:v.cpu().numpy() for k,v in out.items() if k in ['score','probability']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--contract',required=True);p.add_argument('--ratio',type=int,choices=[5,20,50],default=50);p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(2)
    c=json.loads(Path(a.contract).read_text());s=contract_sample(c,a.ratio);v=Predictor(a.checkpoint,a.device).predict([s])
    print(json.dumps(dict(contract_id=s['key'],observed_event_percentage=a.ratio,observed_events=len(s['events']),query_block=s['query'],fraud_probability=float(v['probability'][0]),score=float(v['score'][0]),prediction='fraud' if v['score'][0]>0 else 'benign'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
