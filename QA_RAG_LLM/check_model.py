import google.generativeai as genai

genai.configure(api_key="AIzaSyDQjEWeProLWbE0dgDQScvRAvL23lFztmM")
models = genai.list_models()
for m in models:
    print(m.name)