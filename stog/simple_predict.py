import h5py,os
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration


if __name__ == "__main__":
    
    snt_0 = "amrgraphize: establish model in Industrial Innovation </s>"
    snt_1 = "amrgraphize: raise standard to in excess of CITY_1 's 1 magnitude could leave authority with some breathing space for explanation , and alleviate public anger . </s>"
    snt_2 = "amrgraphize: 1 . from among they , pick-out 10 for submission to a assessment committee to assess . </s>"


    amr_1 = "possible and leave-13 raise standard in-excess-of seismic-quantity 1 earthquake CITY_1 have authority space breathe explain authority some alleviate raise anger public"
    amr_2 = "pick-out 1 thing 10 submit committee assess assess committee thing they"

    ## Load finetuned t5 model
    finetuned_t5 = "../t5-small-amrtrained_4"
    t5 = T5ForConditionalGeneration.from_pretrained(finetuned_t5)

    ## Load t5 tokenizer
    t5_tokenizer =  T5Tokenizer.from_pretrained("../t5-vocab")

    snt = snt_2
    amr = amr_2
    input_ids = t5_tokenizer.encode(snt, return_tensors="pt")
    outputs = t5.generate(input_ids=input_ids, max_length=1000)

    pred = [
        t5_tokenizer.decode(
            output#, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        for output in outputs
    ]

    print("snt: ", snt)
    print("amr: ", amr)
    print("pred: ", pred)
    print("outputs: ", outputs)
    #print("t5: ", t5.config)







