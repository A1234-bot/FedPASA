"""Cache immutable text work; keep the released FedMVP forward unchanged.

Only frozen token embeddings and the fixed zero-shot text branch are cached.
Learned prompt text embeddings, both image passes, loss and RNG are unchanged.
Caches are per-model ordinary attributes and never enter checkpoints/FedAvg.
"""
import types
import torch

VERSION='immutable_text_cache_v1'

def enable(net):
    if getattr(net,'_fedmvp_fast_enabled',False):return
    original_tokens=net.get_tokenized_classnames
    original_fixed=net.clip_model_.encode_text
    tokens={}; fixed={}

    def precision_key():
        return (str(net.token_embedding.weight.device),net.dtype,
                torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'))

    def cached_tokens(self,classnames):
        key=(tuple(classnames),self.prompt_prefix,self.n_ctx,precision_key())
        if key not in tokens:tokens[key]=original_tokens(classnames)
        return tokens[key]

    def cached_fixed(self,text):
        # This method is called with the immutable "a photo of a CLASS." tokens
        # in the release's no_grad branch. Validate exact tokens on cache fill;
        # tokenizer caching below returns the same tensor objects every batch.
        if torch.is_grad_enabled():return original_fixed(text)
        key=precision_key()
        if key not in fixed:
            fixed[key]=(text.detach().clone(),original_fixed(text).detach())
        reference,value=fixed[key]
        # The installed forward always uses the same class names. A different
        # input must not silently get the first input's features.
        if text.shape!=reference.shape or not torch.equal(text,reference):return original_fixed(text)
        return value

    # Tokenization itself is CPU work repeated 80/81 times per batch. Wrap only
    # the release module's tokenizer via a proxy; other CLIP modules unaffected.
    # Each cached CPU token tensor is immutable in the release forward.
    import model.fedmvp as module
    if not hasattr(module,'_fedmvp_original_clip'):
        original_clip=module.clip
        class ClipProxy:
            def __init__(self):self.cache={}
            def __getattr__(self,name):return getattr(original_clip,name)
            def tokenize(self,text,context_length=77,truncate=False):
                key=(text if isinstance(text,str) else tuple(text),context_length,truncate)
                if key not in self.cache:self.cache[key]=original_clip.tokenize(text,context_length,truncate)
                return self.cache[key]
        module._fedmvp_original_clip=original_clip
        module.clip=ClipProxy()
    net.get_tokenized_classnames=types.MethodType(cached_tokens,net)
    net.clip_model_.encode_text=types.MethodType(cached_fixed,net.clip_model_)
    net._fedmvp_fast_enabled=True

def disable_token_proxy():
    import model.fedmvp as module
    if hasattr(module,'_fedmvp_original_clip'):
        module.clip=module._fedmvp_original_clip
        del module._fedmvp_original_clip
