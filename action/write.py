# sentence write template 
from langchain.chains import  ConversationChain
from langchain.prompts import ChatPromptTemplate   
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.schema.output_parser import OutputParserException
from retrying import retry

origin1 = ResponseSchema(name="1.",description="load the origin1")
origin2 = ResponseSchema(name="2.",description="load the origin2")
origin3 = ResponseSchema(name="3.",description="load the origin3")
origin4 = ResponseSchema(name="4.",description="load the origin4")
origin5 = ResponseSchema(name="5.",description="load the origin5")
origin6 = ResponseSchema(name="6.",description="load the origin6")
origin7 = ResponseSchema(name="7.",description="load the origin7")
origin8 = ResponseSchema(name="8.",description="load the origin8")
origin9 = ResponseSchema(name="9.",description="load the origin9")
origin10 = ResponseSchema(name="10.",description="load the origin10")

change1 = ResponseSchema(name="Change1:",description="Change1:")
change2 = ResponseSchema(name="Change2:",description="Change2:")
change3 = ResponseSchema(name="Change3:",description="Change3:")
change4 = ResponseSchema(name="Change4:",description="Change4:")
change5 = ResponseSchema(name="Change5:",description="Change5:")
change6 = ResponseSchema(name="Change6:",description="Change6:")
change7 = ResponseSchema(name="Change7:",description="Change7:")
change8 = ResponseSchema(name="Change8:",description="Change8:")
change9 = ResponseSchema(name="Change9:",description="Change9:")
change10 = ResponseSchema(name="Change10:",description="Change10:")

write_schemas = [origin1, change1, origin2, change2, origin3, change3, origin4, change4, origin5, change5, origin6, change6, origin7, change7, origin8, change8, origin9, change9, origin10, change10]
write_parser = StructuredOutputParser.from_response_schemas(write_schemas)
write_format_instructions = write_parser.get_format_instructions()



def retry_if_output_parser_exception(exception):
    return isinstance(exception, OutputParserException)

@retry(wait_random_min=256, wait_random_max=1000*10, stop_max_attempt_number=5, retry_on_exception=retry_if_output_parser_exception)
def write_sent(llm, raw_data):
    conversation_write = ConversationChain(llm=llm, verbose=True)
    write_template ='''
        Role:
        <<<You are a good sentence changer. Now I want you to help make a dataset containing sentence perturbations. Learning from examples, do the rest cases >>>
    
        Examples:
        <<<      
        1. "The dog chases the cat."
        Change: "The cat chases the dog."

        2. "She heard he talked to the teacher"
        Change: "He heard she talked to the teacher" 

        3. "Curiosity fuels a lifelong passion for learning."
        Change: "A lifelong passion for learning fuels curiosity."

        4. "The bond market reacted to economic data."
        Change: "Economic data reacted to the bond market." 
        >>>

        Demand: 
        <<<Learn from these examples; they feature subtle syntax changes (word order) that result in significant semantic differences (relation changes). Please adhere to this rule.>>>
        Voice change like "ride" -> "be/ being riden" is not allowed.

        Your Cases:
        Origin1. {sent1}
        Origin2. {sent2}
        Origin3. {sent3}
        Origin4. {sent4}
        Origin5. {sent5}
        Origin6. {sent6}
        Origin7. {sent7}
        Origin8. {sent8}
        Origin9. {sent9}
        Origin10. {sent10}
        <<<{format_instructions}>>>

        '''
    #write 
    write_prompt = ChatPromptTemplate.from_template(template=write_template)
    prompt_w1 = write_prompt.format_messages(sent1 = raw_data[0], sent2 = raw_data[1], sent3 = raw_data[2],\
        sent4 = raw_data[3], sent5 = raw_data[4],sent6 = raw_data[5], sent7 = raw_data[6], sent8 = raw_data[7],\
        sent9 = raw_data[8], sent10 = raw_data[9],format_instructions=write_format_instructions)
    ans1 = conversation_write.predict(input = prompt_w1[0].content)
    ans1_dict = write_parser.parse(ans1)
    # ans1_dict = parse_ans1(ans1)
    # print(ans1)
    return ans1_dict

