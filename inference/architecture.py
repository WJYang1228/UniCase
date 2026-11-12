
import math
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
torch.backends.mha.set_fastpath_enabled(False)


def mlp(a,h,b,dropout):
    return nn.Sequential(nn.Linear(a,h),nn.LayerNorm(h),nn.GELU(),nn.Dropout(dropout),nn.Linear(h,b))


class DistanceHead(nn.Module):
    def __init__(self,c,state):
        super().__init__();self.c=c
        self.projector=mlp(2*c['projection_dim'],c['hidden'],c['space_dim'],c['dropout'])
        width=state['head.maps.0.0.weight'].shape[0]
        self.maps=nn.ModuleList([mlp(c['space_dim'],width,c['space_dim'],c['dropout'])
                                for _ in range(1 if c['shared_mapping'] else 2)])
        self.anchors=nn.Parameter(torch.empty(2,c['anchors'],c['space_dim']))
        self.raw_alpha=nn.Parameter(torch.empty(()))
    def forward(self,joint):
        shared=self.projector(joint);ds=[];c=self.c
        for label in (0,1):
            z=F.normalize(self.maps[0 if c['shared_mapping'] else label](shared).float(),dim=-1)
            a=F.normalize(self.anchors[label].float(),dim=-1)
            delta=(z[:,None]-a[None]).square().sum(-1)
            if c['aggregation']=='softmin':
                raw=-c['anchor_temperature']*(torch.logsumexp(-delta/c['anchor_temperature'],dim=-1)-math.log(c['anchors']))
            elif c['aggregation']=='weighted':
                raw=((-delta/c['anchor_temperature']).softmax(-1)*delta).sum(-1)
            else:raise ValueError('Unsupported aggregation')
            denominator=(torch.pdist(a,p=2).square().mean().detach()+c['epsilon']) if c['normalize_anchor_scale'] else raw.new_ones(())
            ds.append(raw/denominator)
        score=ds[0]-ds[1];alpha=F.softplus(self.raw_alpha)+torch.finfo(self.raw_alpha.dtype).tiny
        return {'score':score,'logits':alpha*score,'probability':(alpha*score).sigmoid()}


class InferenceModel(nn.Module):
    def __init__(self,c,state):
        super().__init__();self.c=c
        self.static_encoder=mlp(c['static_dim'],c['hidden'],c['hidden'],c['dropout'])
        self.event_projection=nn.Linear(c['event_dim'],c['hidden'])
        layer=nn.TransformerEncoderLayer(c['hidden'],c['attention_heads'],4*c['hidden'],c['dropout'],
                                        activation='gelu',batch_first=True,norm_first=True)
        self.dynamic_encoder=nn.TransformerEncoder(layer,c['transformer_layers'],enable_nested_tensor=False)
        self.empty_context=nn.Parameter(torch.empty(c['hidden']))
        self.project_static=mlp(c['hidden'],c['hidden'],c['projection_dim'],c['dropout'])
        self.project_dynamic=mlp(c['hidden'],c['hidden'],c['projection_dim'],c['dropout'])
        self.head=DistanceHead(c,state) if c['classifier']=='dual' else mlp(2*c['projection_dim'],state['head.0.weight'].shape[0],1,c['dropout'])
        self.load_state_dict(state,strict=True)

    def forward(self,static,events,mask):
        if static.ndim!=2 or events.ndim!=3 or mask.shape!=events.shape[:2] or mask.dtype!=torch.bool:
            raise ValueError('Invalid input dimensions or mask')
        if not events.shape[1] or torch.any(mask[:,1:] & ~mask[:,:-1]):raise ValueError('Invalid prefix mask')
        c=self.c;hs=self.static_encoder(static);lengths=mask.sum(1)
        hd=self.empty_context.unsqueeze(0).expand(len(static),-1).clone();valid=lengths>0
        if valid.any():
            x=self.event_projection(events[valid].masked_fill(~mask[valid,:,None],0))
            pos=torch.arange(x.shape[1],device=x.device,dtype=x.dtype)[:,None]
            omega=torch.exp(torch.arange(0,c['hidden'],2,device=x.device,dtype=x.dtype)*(-math.log(10000)/c['hidden']))
            pe=torch.zeros_like(x[0]);pe[:,0::2]=torch.sin(pos*omega);pe[:,1::2]=torch.cos(pos*omega)
            context=self.dynamic_encoder(x+pe,src_key_padding_mask=~mask[valid])
            hd[valid]=context[torch.arange(valid.sum(),device=x.device),lengths[valid]-1]
        joint=torch.cat((self.project_static(hs),self.project_dynamic(hd)),dim=-1)
        if c['classifier']=='dual':return self.head(joint)
        logits=self.head(joint).squeeze(-1)
        return {'score':logits,'logits':logits,'probability':logits.sigmoid()}


class Predictor:
    def __init__(self,checkpoint,device='cpu'):
        obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if obj.get('version')!='unicast-core-revision-20261111-v1' or not obj.get('inference_only'):
            raise ValueError('Unsupported checkpoint format')
        for name,value in obj['state_dict'].items():
            if not torch.isfinite(value).all():raise ValueError('Nonfinite weight '+name)
        self.normalization={k:v.numpy() for k,v in obj['normalization'].items()}
        for k,n in [('mean_static',264),('std_static',264),('mean_event',16),('std_event',16)]:
            a=self.normalization[k]
            if a.shape!=(n,) or not np.isfinite(a).all() or (k.startswith('std_') and np.any(a<=0)):
                raise ValueError('Invalid normalization '+k)
        self.model=InferenceModel(obj['config'],obj['state_dict']).to(device).eval();self.device=device

    @torch.inference_mode()
    def predict_features(self,static_raw,event_raw):
        n=self.normalization
        static=(np.asarray(static_raw,np.float32)-n['mean_static'])/n['std_static']
        length=max(1,max(len(x) for x in event_raw));events=np.zeros((len(event_raw),length,16),np.float32)
        mask=np.zeros((len(event_raw),length),bool)
        for i,x in enumerate(event_raw):
            if len(x):events[i,:len(x)]=(np.asarray(x)-n['mean_event'])/n['std_event'];mask[i,:len(x)]=True
        values=self.model(*(torch.as_tensor(x,device=self.device) for x in (static,events,mask)))
        return {k:v.cpu().numpy() for k,v in values.items()}
