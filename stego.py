"""Research prototype: adaptive LSB, DCT-QIM, and an optional learned image steganography channel.

The project embeds benign UTF-8 data and measures extraction, image distortion and
processing robustness. It does NOT optimize for changing a VLM's instructions.

Example:
  python stego.py demo --output demo
  python stego.py encode --input cover.png --message 'experiment 42' --output stego.png --meta stego.json
  python stego.py decode --input stego.png --meta stego.json
  python stego.py train-neural --checkpoint neural.pt --steps 300
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from scipy.fft import dctn, idctn

# Encoded bits contain 4-byte length + 8-byte integrity check + payload.
def pack_message(message: str) -> np.ndarray:
    data = message.encode('utf-8')
    if len(data) > 2**32 - 1:
        raise ValueError('Payload too large')
    header = len(data).to_bytes(4, 'big') + hashlib.sha256(data).digest()[:8]
    return np.unpackbits(np.frombuffer(header + data, dtype=np.uint8))


def unpack_message(bits: np.ndarray) -> str:
    raw = np.packbits(bits.astype(np.uint8)).tobytes()
    if len(raw) < 12:
        raise ValueError('Payload is truncated')
    size = int.from_bytes(raw[:4], 'big')
    if 12 + size > len(raw):
        raise ValueError('Invalid length: corrupted bitstream')
    data = raw[12:12 + size]
    if hashlib.sha256(data).digest()[:8] != raw[4:12]:
        raise ValueError('Integrity mismatch: extraction failed')
    return data.decode('utf-8')


def local_complexity(image: np.ndarray) -> np.ndarray:
    """Simple 3x3 grayscale local variation and gradient magnitude."""
    from scipy.ndimage import uniform_filter, sobel
    gray = image.astype(np.float32).mean(axis=2)
    mu = uniform_filter(gray, 3, mode='reflect')
    var = np.maximum(uniform_filter(gray**2, 3, mode='reflect') - mu**2, 0)
    grad = np.hypot(sobel(gray, axis=0), sobel(gray, axis=1))
    return np.sqrt(var) + 0.25 * grad


def embed_lsb(image: np.ndarray, bits: np.ndarray, key: int, max_depth: int = 3):
    """Adaptive depth on pseudorandomly permuted pixel/channel positions.

    The sidecar records positions and depths: this is NON-BLIND extraction.
    It avoids deriving the decoder's selection map from a modified image.
    """
    if image.ndim != 3 or image.dtype != np.uint8:
        raise ValueError('Expected uint8 HxWxC image')
    out = image.copy()
    H, W, C = out.shape
    score = local_complexity(image)
    low, high = np.quantile(score, [0.40, 0.80])
    depths = np.where(score > high, 3, np.where(score > low, 2, 1))
    depths = np.minimum(depths, max_depth)
    rng = np.random.default_rng(key)
    order = rng.permutation(H * W * C)
    flat = out.reshape(-1)
    depth_flat = np.repeat(depths.reshape(-1), C)
    records = []
    consumed = 0
    for index in order:
        if consumed == len(bits):
            break
        d = min(int(depth_flat[index]), len(bits) - consumed)
        value = int(np.packbits(np.pad(bits[consumed:consumed+d], (0, 8-d)))[0]) >> (8-d)
        flat[index] = (int(flat[index]) & (~((1 << d)-1) & 255)) | value
        records.append([int(index), d])
        consumed += d
    if consumed != len(bits):
        raise ValueError('LSB region too small for payload')
    return out, records


def extract_lsb(image: np.ndarray, records: list[list[int]]) -> np.ndarray:
    flat = image.reshape(-1)
    out = []
    for idx, depth in records:
        value = int(flat[idx]) & ((1 << depth) - 1)
        out.extend((value >> j) & 1 for j in range(depth-1, -1, -1))
    return np.asarray(out, dtype=np.uint8)


# Fixed middle-frequency slots in an 8x8 block. DCT-QIM embeds one bit/block.
DCT_SLOTS = [(2, 3), (3, 2), (1, 4), (4, 1)]

def embed_dct(image: np.ndarray, bits: np.ndarray, q: float = 24.0):
    """Parity quantization-index modulation on luma (YCbCr Y plane).

    q is a coefficient quantization step; increasing q improves recoverability
    but generally worsens image distortion. No JPEG survivability is guaranteed.
    """
    pil = Image.fromarray(image, 'RGB').convert('YCbCr')
    arr = np.asarray(pil, dtype=np.uint8).copy()
    H, W, _ = arr.shape
    ncols = W // 8
    nblocks = (H // 8) * ncols
    if len(bits) > nblocks:
        raise ValueError(f'DCT capacity {nblocks} bits, need {len(bits)}')
    y = arr[:,:,0].astype(np.float64)
    for i, bit in enumerate(bits):
        r, c = divmod(i, ncols)
        sl = (slice(r*8, (r+1)*8), slice(c*8, (c+1)*8))
        F = dctn(y[sl], norm='ortho')
        u,v = DCT_SLOTS[i % len(DCT_SLOTS)]
        z = F[u,v] / q
        k = int(np.round(z))
        if (k % 2) != int(bit):
            k = k-1 if abs((k-1)-z) < abs((k+1)-z) else k+1
        F[u,v] = k*q
        y[sl] = idctn(F, norm='ortho')
    arr[:,:,0] = np.clip(np.rint(y), 0, 255).astype(np.uint8)
    result = np.asarray(Image.fromarray(arr, 'YCbCr').convert('RGB'), dtype=np.uint8)
    return result


def extract_dct(image: np.ndarray, nbits: int, q: float = 24.0) -> np.ndarray:
    y = np.asarray(Image.fromarray(image, 'RGB').convert('YCbCr'), dtype=np.float64)[:,:,0]
    H, W = y.shape
    ncols = W // 8
    if nbits > (H//8)*ncols:
        raise ValueError('Not enough DCT blocks')
    bits = []
    for i in range(nbits):
        r,c = divmod(i,ncols)
        F = dctn(y[r*8:(r+1)*8,c*8:(c+1)*8], norm='ortho')
        u,v = DCT_SLOTS[i % len(DCT_SLOTS)]
        bits.append(int(np.rint(F[u,v]/q)) % 2)
    return np.asarray(bits, dtype=np.uint8)


def segment_regions(width: int, weights: tuple[float,float,float], neural: bool):
    """Two/three vertical, 8-aligned segments; weight sets physical capacity allocation."""
    a,b,g = weights
    if min(a,b,g) < 0 or abs(a+b+g-1) > 1e-5:
        raise ValueError('Weights must be nonnegative and sum to one')
    if not neural:
        a,b = a/(a+b),b/(a+b)
        edges = [0, max(8, min(width-8, int(width*a)//8*8)), width]
        return edges
    if width < 192:
        raise ValueError('Neural hybrid requires image width >= 192')
    e1 = max(64,int(width*a)//8*8)
    e2 = max(e1+64,int(width*(a+b))//8*8)
    e2 = min(width-64, e2)
    if e1 >= e2:
        raise ValueError('Image too narrow for these weights')
    return [0,e1,e2,width]


def split_bits(bits: np.ndarray, weights, neural=False):
    """Payload proportions. Enforce >= 1 bit per enabled channel when possible."""
    w = np.asarray(weights if neural else weights[:2], dtype=float)
    w /= w.sum()
    counts = np.floor(len(bits)*w).astype(int)
    order = np.argsort(-(len(bits)*w-counts))
    for i in range(len(bits)-counts.sum()):
        counts[order[i]] += 1
    return np.split(bits, np.cumsum(counts)[:-1]), counts.tolist()


def encode(image: np.ndarray, message: str, key: int=17, q:float=24,
           weights=(.45,.35,.20), neural_model=None):
    """Hybrid codec; PNG output required for exact extraction.

    Learned channel encodes at most its model's fixed bit length; zero-padding is used.
    """
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('Supply 8-bit RGB image')
    bits = pack_message(message)
    edges = segment_regions(image.shape[1], weights, neural_model is not None)
    parts, counts = split_bits(bits, weights, neural_model is not None)
    result = image.copy()
    result[:,edges[0]:edges[1]], records = embed_lsb(result[:,edges[0]:edges[1]],parts[0],key)
    result[:,edges[1]:edges[2]] = embed_dct(result[:,edges[1]:edges[2]],parts[1],q)
    meta = {'version':1,'edges':edges,'counts':counts,'lsb_records':records,
            'dct_q':q,'neural': neural_model is not None}
    if neural_model is not None:
        import torch
        n = neural_model.message_bits
        if len(parts[2]) > n:
            raise ValueError(f'Neural region requires <= {n} bits, got {len(parts[2])}')
        sub = result[:,edges[2]:edges[3]]
        if sub.shape[0] != 64 or sub.shape[1] != 64:
            raise ValueError('Neural region must be 64x64 (resize/crop image beforehand)')
        padded = np.pad(parts[2],(0,n-len(parts[2])))
        with torch.no_grad():
            x=torch.from_numpy(sub.copy()).permute(2,0,1).unsqueeze(0).float()/255.
            m=torch.from_numpy(padded.astype(np.float32)).unsqueeze(0)
            stego=neural_model.encoder(x,m).clamp(0,1)
            result[:,edges[2]:edges[3]]=(stego[0].permute(1,2,0).numpy()*255).round().astype(np.uint8)
    return result,meta


def decode(image: np.ndarray, meta: dict, neural_model=None) -> str:
    e=meta['edges']; n=meta['counts']
    lsb=extract_lsb(image[:,e[0]:e[1]],meta['lsb_records'])
    dct=extract_dct(image[:,e[1]:e[2]],n[1],meta['dct_q'])
    chunks=[lsb,dct]
    if meta['neural']:
        if neural_model is None: raise ValueError('Neural checkpoint required')
        import torch
        with torch.no_grad():
            sub=image[:,e[2]:e[3]]
            x=torch.from_numpy(sub.copy()).permute(2,0,1).unsqueeze(0).float()/255.
            logits=neural_model.decoder(x)
            chunks.append((logits[0,:n[2]] > 0).numpy().astype(np.uint8))
    return unpack_message(np.concatenate(chunks))


def stats(original: np.ndarray, modified: np.ndarray):
    diff=original.astype(float)-modified.astype(float)
    mse=float(np.mean(diff**2))
    return {'mse':mse,'psnr_db':float('inf') if mse==0 else float(10*np.log10(255**2/mse)),
            'mean_abs_error':float(np.mean(np.abs(diff))),
            'fraction_pixels_changed':float(np.mean(np.any(original != modified,axis=2)))}


def test_transforms(image: np.ndarray, meta: dict, original_message: str, neural_model=None):
    """Evaluate rather than claim robustness. Lossy transformations may fail integrity check."""
    variants={'png_identity':image,
              'jpeg_95':np.asarray(_jpeg(image,95)),
              'jpeg_80':np.asarray(_jpeg(image,80)),
              'gaussian_blur_0.5':np.asarray(Image.fromarray(image).filter(ImageFilter.GaussianBlur(.5)))}
    report={}
    for name,variant in variants.items():
        try:
            recovered=decode(variant,meta,neural_model)
            report[name]={'exact_recovery':recovered==original_message,'message':recovered}
        except Exception as e:
            report[name]={'exact_recovery':False,'error':str(e)}
    return report


def _jpeg(image,quality):
    out=io.BytesIO()
    Image.fromarray(image).save(out,format='JPEG',quality=quality)
    out.seek(0)
    return Image.open(out).convert('RGB')


# Optional trained channel. Training demo below uses synthetic smooth images,
# not a benchmark dataset. A neural method must be trained before useful decoding.
def make_neural_model(message_bits: int=32, strength: float=.08):
    import torch
    from torch import nn
    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Conv2d(3+message_bits,48,3,padding=1),nn.ReLU(),
                nn.Conv2d(48,48,3,padding=1),nn.ReLU(),
                nn.Conv2d(48,3,3,padding=1),nn.Tanh())
        def forward(self,x,m):
            plane=m[:,:,None,None].expand(-1,-1,x.shape[2],x.shape[3])
            return x + strength*self.net(torch.cat([x,plane],dim=1))
    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Conv2d(3,32,3,padding=1),nn.ReLU(),
                nn.Conv2d(32,64,3,stride=2,padding=1),nn.ReLU(),
                nn.Conv2d(64,64,3,stride=2,padding=1),nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(64,message_bits))
        def forward(self,x):return self.net(x)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.message_bits=message_bits
            self.encoder=Encoder();self.decoder=Decoder()
    return Model()


def train_neural(checkpoint,steps=300,batch_size=16,seed=17):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    model=make_neural_model();optim=torch.optim.Adam(model.parameters(),lr=1e-3)
    for step in range(steps):
        # Spatially smoothed synthetic images, ranging from low to high texture.
        base=torch.rand(batch_size,3,16,16)
        x=F.interpolate(base,size=(64,64),mode='bilinear',align_corners=False)
        x=(x*.8+.1).clamp(0,1)
        bits=torch.randint(0,2,(batch_size,model.message_bits)).float()
        stego=model.encoder(x,bits).clamp(0,1)
        # Approximate mild image perturbations in some batches.
        noisy=(stego+0.005*torch.randn_like(stego)).clamp(0,1)
        logits=model.decoder(noisy)
        recovery=F.binary_cross_entropy_with_logits(logits,bits)
        perceptual=F.mse_loss(stego,x)
        residual=(stego-x).abs().mean()
        loss=recovery+5*perceptual+0.5*residual
        optim.zero_grad();loss.backward();optim.step()
        if (step+1)%max(1,steps//5)==0:
            acc=((logits>0)==bits.bool()).float().mean().item()
            print(f'step={step+1} loss={loss.item():.4f} bit_accuracy={acc:.3f}')
    torch.save({'state':model.state_dict(),'bits':model.message_bits},checkpoint)


def load_neural(checkpoint):
    import torch
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=make_neural_model(data['bits'])
    model.load_state_dict(data['state']);model.eval()
    return model


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('demo');p.add_argument('--output',default='demo')
    p=sub.add_parser('encode');p.add_argument('--input',required=True);p.add_argument('--message',required=True)
    p.add_argument('--output',default='stego.png');p.add_argument('--meta',default='stego.json')
    p.add_argument('--key',type=int,default=17);p.add_argument('--q',type=float,default=24)
    p.add_argument('--image-type',choices=['natural','synthetic','document'],default='natural')
    p.add_argument('--neural-checkpoint')
    p=sub.add_parser('decode');p.add_argument('--input',required=True);p.add_argument('--meta',required=True)
    p.add_argument('--neural-checkpoint')
    p=sub.add_parser('train-neural');p.add_argument('--checkpoint',default='neural.pt')
    p.add_argument('--steps',type=int,default=300);p.add_argument('--batch-size',type=int,default=16)
    args=parser.parse_args()
    if args.mode=='train-neural':
        train_neural(args.checkpoint,args.steps,args.batch_size);return
    if args.mode=='decode':
        image=np.asarray(Image.open(args.input).convert('RGB'))
        meta=json.loads(Path(args.meta).read_text())
        model=load_neural(args.neural_checkpoint) if args.neural_checkpoint else None
        print(decode(image,meta,model));return
    weights={'natural':(.45,.35,.20),'synthetic':(.30,.40,.30),'document':(.25,.50,.25)}
    if args.mode=='demo':
        rng=np.random.default_rng(17)
        x,y=np.meshgrid(np.arange(384),np.arange(256))
        arr=np.stack([80+60*np.sin(x/20)+20*np.cos(y/10),
                      110+50*np.cos(x/17+y/35),
                      150+40*np.sin(y/14)],axis=-1)
        image=np.clip(arr+rng.normal(0,9,arr.shape),0,255).astype(np.uint8)
        message='Hello, this is a benign hidden message.'
        prefix=args.output;out=prefix+'_stego.png';meta_path=prefix+'_meta.json'
        model=None;chosen=weights['natural'];key=17;q=24
        Image.fromarray(image).save(prefix+'_cover.png')
    else:
        image=np.asarray(Image.open(args.input).convert('RGB'))
        message=args.message;out=args.output;meta_path=args.meta
        model=load_neural(args.neural_checkpoint) if args.neural_checkpoint else None
        chosen=weights[args.image_type];key=args.key;q=args.q
    result,meta=encode(image,message,key=key,q=q,weights=chosen,neural_model=model)
    Image.fromarray(result).save(out)
    Path(meta_path).write_text(json.dumps(meta))
    print(json.dumps({'input_shape':list(image.shape),'message_bits':len(pack_message(message)),
        'channel_bits':meta['counts'],'quality':stats(image,result),
        'robustness':test_transforms(result,meta,message,model)},indent=2))
    print('saved:',out,meta_path)

if __name__=='__main__':main()
