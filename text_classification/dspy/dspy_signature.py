import dspy

'''
Task class definition
The comments in class are prompt
'''

class USCN_comment_detection_CN2US(dspy.Signature):
    """你需要对来自中国的短视频平台上的评论进行情感分析。
    请将评论分为3类，分别是：1.仇恨、攻击性言论以及批评和抱怨，2.相对中立的评论，3.支持和认可。
    """

    短视频题目: str = dspy.InputField(desc='这是视频的标题') 
    评论: str = dspy.InputField(desc='这是你需要分析的视频评论区的评论') 
    ip地址: str = dspy.InputField(desc='这是发表评论的用户的ip地址') 

    情感分类: int = dspy.OutputField(desc='请给出评论对美国的情感态度的分类，1为仇恨、攻击性言论以及批评和抱怨，2为相对中立的评论，3为支持和认可') 

class USCN_comment_detection_CN2US_dspy(dspy.Signature):
    """请对来自中国的短视频平台上的评论进行情感分类。
    请将评论分为3类，分别是：1.仇恨、攻击性言论以及批评和抱怨，2.相对中立的评论，3.支持和认可。请根据评论内容和情感倾向进行分析，并提供相应的分类结果。
    """

    短视频题目: str = dspy.InputField(desc='这是视频的标题') 
    评论: str = dspy.InputField(desc='这是你需要分析的视频评论区的评论') 
    ip地址: str = dspy.InputField(desc='这是发表评论的用户的ip地址') 

    情感分类: int = dspy.OutputField(desc='请给出评论对美国的情感态度的分类，1为仇恨、攻击性言论以及批评和抱怨，2为相对中立的评论，3为支持和认可') 


class USCN_comment_detection_US2CN(dspy.Signature):
    """You need to do sentiment analysis of comments on short video platforms from the United States.
       Please categorize comments into 3 categories: 1. Hate, offensive speech and criticism and complaints, 2. Relatively neutral comments, 3. Support and approval.
    """

    Short_video_title: str = dspy.InputField(desc='Here is the title of the video') 
    comment: str = dspy.InputField(desc='Here is the comment in the comments section of the video you need to analyze') 

    sentiment_category: int = dspy.OutputField(desc='Please give a category of emotional attitudes towards China, 1 as hate, offensive speech, criticism and complaint, 2 as relatively neutral comments, 3 as support and approval') 

class USCN_comment_detection_US2CN_dspy(dspy.Signature):
    """Analyze the sentiment of comments on short video platforms related to Asian and Chinese culture. 
    Classify each comment into one of the following 3 categories: 1. Hate, offensive speech and criticism and complaints, 2. Relatively neutral comments, 3. Support and approval. 
    Focus on identifying the nuances in cultural engagement and the audience's openness to different cultural perspectives.",
    """

    Short_video_title: str = dspy.InputField(desc='Here is the title of the video') 
    comment: str = dspy.InputField(desc='Here is the comment in the comments section of the video you need to analyze') 

    sentiment_category: int = dspy.OutputField(desc='Please give a category of emotional attitudes towards China, 1 as hate, offensive speech, criticism and complaint, 2 as relatively neutral comments, 3 as support and approval') 

