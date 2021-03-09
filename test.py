from deeppavlov import build_model
 
model = build_model('models/squad_torch_bert.json', download=True)
model("When was James born?","James was born in 1977.")
model(["James was born in 1977."],["When was James born?"]) 
