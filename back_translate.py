import http.client
import hashlib
import urllib
import random
import json
import time

def baidu_translate(content,from_lang,to_lang):
    appid = 'CqQCtGjG1Y81KCGKR0TnUh6X'
    secretKey = 'o8oSRkl1ZiInsh2ctzGqGSWKRKQ7qnID'
    httpClient = None
    myurl = '/api/trans/vip/translate'
    q = content
    fromLang = from_lang
    toLang = to_lang
    salt = random.randint(32768, 65536)
    sign = appid + q + str(salt) + secretKey
    sign = hashlib.md5(sign.encode()).hexdigest()
    myurl = myurl + '?appid=' + appid + '&q=' + urllib.parse.quote(
        q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
        salt) + '&sign=' + sign
    try:
        httpClient = http.client.HTTPConnection('api.fanyi.baidu.com')
        httpClient.request('GET', myurl)
        response = httpClient.getresponse()
        jsonResponse = response.read().decode("utf-8")
        js = json.loads(jsonResponse)
        dst = str(js["trans_result"][0]["dst"])
        return dst
    except Exception as e:
        print('err:',e)
    finally:
        if httpClient:
            httpClient.close()
 
 
def do_translate(content,from_lang,to_lang):
    if len(content)>= 260:
        content = content[0:260]
    temp = baidu_translate(content,from_lang,to_lang)
    time.sleep(10)
    if temp is None:
        temp = 0
    if len(temp) >= 1500:
        temp = temp[0:1500]
    res = baidu_translate(temp,to_lang,from_lang)
    return res


def get_back_translate_df(sentences):
    results = []
    for i, sents in enumerate(sentences):
        augmented_sentences = do_translate(do_translate(sents,'en','zh'),'zh','en')
        results.append(augmented_sentences)
    return results