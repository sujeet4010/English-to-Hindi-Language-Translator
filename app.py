import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TFAutoModelForSeq2SeqLM

if __name__ == "__main__":
    # Load the model once when the app starts
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

    # Load the saved model
    model = TFAutoModelForSeq2SeqLM.from_pretrained("tf_model/")
    model_checkpoint = "Helsinki-NLP/opus-mt-en-hi"
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load the saved GenerationConfig
    generation_config = GenerationConfig.from_pretrained('tf_model/')
    
    st.title("Text Generation with Hugging Face Transformers")
    input_text = st.text_area("Enter text for inference", height=150)
    
    
    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
    # Perform inference using the model and generation configuration
    outputs = model.generate(input_ids, 
                            max_length=generation_config.max_length, 
                            num_beams=generation_config.num_beams, 
                            bad_words_ids=generation_config.bad_words_ids)

    # Decode the generated output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.title("Text Generated after the translation")
    st.write(generated_text)
    # Print the result
    print("Generated text:", generated_text)

