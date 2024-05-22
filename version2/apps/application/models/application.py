# coding=utf-8
"""
    @project: maxkb
    @Author：虎
    @file： application.py
    @date：2023/9/25 14:24
    @desc:
"""
import uuid

from django.contrib.postgres.fields import ArrayField
from django.db import models
from langchain.schema import HumanMessage, AIMessage

from common.mixins.app_model_mixin import AppModelMixin
from dataset.models.data_set import DataSet
from setting.models.model_management import Model
from users.models import User


def get_dataset_setting_dict():
    return {'top_n': 3, 'similarity': 0.6, 'max_paragraph_char_number': 5000, 'search_mode': 'embedding',
            'no_references_setting': {
                'status': 'ai_questioning',
                'value': '{question}'
            }}


def get_model_setting_dict():
    return {'prompt': Application.get_default_model_prompt()}


class Application(AppModelMixin):
    id = models.UUIDField(primary_key=True, max_length=128, default=uuid.uuid1, editable=False, verbose_name="主键id")
    name = models.CharField(max_length=128, verbose_name="应用名称")
    desc = models.CharField(max_length=512, verbose_name="引用描述", default="")
    prologue = models.CharField(max_length=1024, verbose_name="开场白", default="")
    dialogue_number = models.IntegerField(default=0, verbose_name="会话数量")
    user = models.ForeignKey(User, on_delete=models.DO_NOTHING)
    model = models.ForeignKey(Model, on_delete=models.SET_NULL, db_constraint=False, blank=True, null=True)
    dataset_setting = models.JSONField(verbose_name="数据集参数设置", default=get_dataset_setting_dict)
    model_setting = models.JSONField(verbose_name="模型参数相关设置", default=get_model_setting_dict)
    problem_optimization = models.BooleanField(verbose_name="问题优化", default=False)
    icon = models.CharField(max_length=256, verbose_name="应用icon", default="/ui/favicon.ico")

    @staticmethod
    def get_default_model_prompt():
        return ('Context is：'
                '\n{data}'
                '\nThe requirement of your answer：'
                '\n- If you have no answer，please answer “No information is found in the context”.'
                '\n- Avoid to use <data></data>.'
                '\n- Align your answer with the information of <data></data>.'
                '\n- Format your answer with Markdown grammar.'
                '\n- Directly return the information if there are picture link, hyperlink or script in <data></data>.'
                '\n- Please answer in English and do not use any other language.'
                '\n Now the question is：'
                '\n{question}')

    class Meta:
        db_table = "application"


class ApplicationDatasetMapping(AppModelMixin):
    id = models.UUIDField(primary_key=True, max_length=128, default=uuid.uuid1, editable=False, verbose_name="主键id")
    application = models.ForeignKey(Application, on_delete=models.CASCADE)
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)

    class Meta:
        db_table = "application_dataset_mapping"


class Chat(AppModelMixin):
    id = models.UUIDField(primary_key=True, max_length=128, default=uuid.uuid1, editable=False, verbose_name="主键id")
    application = models.ForeignKey(Application, on_delete=models.CASCADE)
    abstract = models.CharField(max_length=1024, verbose_name="Summary")
    client_id = models.UUIDField(verbose_name="Client id", default=None, null=True)

    class Meta:
        db_table = "application_chat"


class VoteChoices(models.TextChoices):
    """订单类型"""
    UN_VOTE = -1, 'NA'
    STAR = 0, 'Agree'
    TRAMPLE = 1, 'Object'


class ChatRecord(AppModelMixin):
    """
    对话日志 详情
    """
    id = models.UUIDField(primary_key=True, max_length=128, default=uuid.uuid1, editable=False, verbose_name="主键id")
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE)
    vote_status = models.CharField(verbose_name='Vote', max_length=10, choices=VoteChoices.choices,
                                   default=VoteChoices.UN_VOTE)
    problem_text = models.CharField(max_length=1024, verbose_name="Question")
    answer_text = models.CharField(max_length=40960, verbose_name="Answer")
    message_tokens = models.IntegerField(verbose_name="Nos of request tokens", default=0)
    answer_tokens = models.IntegerField(verbose_name="Nos of response tokens", default=0)
    const = models.IntegerField(verbose_name="Total Cost", default=0)
    details = models.JSONField(verbose_name="Q&A Details", default=dict)
    improve_paragraph_id_list = ArrayField(verbose_name="改进标注列表",
                                           base_field=models.UUIDField(max_length=128, blank=True)
                                           , default=list)
    run_time = models.FloatField(verbose_name="Runtime", default=0)
    index = models.IntegerField(verbose_name="对话下标")

    def get_human_message(self):
        if 'problem_padding' in self.details:
            return HumanMessage(content=self.details.get('problem_padding').get('padding_problem_text'))
        return HumanMessage(content=self.problem_text)

    def get_ai_message(self):
        return AIMessage(content=self.answer_text)

    class Meta:
        db_table = "application_chat_record"
