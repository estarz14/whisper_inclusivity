from .expert import UpstreamExpert as _UpstreamExpert

def whisper_pt(*args, **kwargs):
    return _UpstreamExpert(variant="pt", mask=False, **kwargs)

def whisper_ft(*args, **kwargs):
    return _UpstreamExpert(variant="ft", mask=False, **kwargs)

# manually change variant & gvd
def whisper_pt_mask(*args, **kwargs):
    print("use mask pt")
    return _UpstreamExpert(variant="pt", mask=True,**kwargs)

def whisper_ft_mask(*args, **kwargs):
    print("use mask ft")
    return _UpstreamExpert(variant="ft", mask=True,**kwargs)

