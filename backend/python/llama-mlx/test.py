from main import BackendServicer

servicer = BackendServicer()
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
servicer.LoadModel(dotdict({
    "ModelFile": "/Users/muriloaraujo/Projetos/Muritavo/LocalAI/models/codellama-mlx-7b"
}), {})

servicer.Predict(dotdict({
    "Prompt": "Work",
    "Tokens": 200,
    "Temperature": 0.9
}), {})