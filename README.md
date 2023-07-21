# ⚖️ 법원 판결 예측 AI 모델 개발
#### 미국 대법원 사례의 사건 식별자와 사건을 분석하여 첫 번째 당사자와 두 번째 당사자 중 첫 번째 당사자의 승소 여부 예측 모델 개발

## 1. 모델 비교
### TfidfVectorizer + LogisticRegression + RandomSearch
```Python
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
def get_vector(vectorizer, df, train_mode):
    if train_mode:
        X_facts = vectorizer.fit_transform(df['facts'])
    else:
        X_facts = vectorizer.transform(df['facts'])
    X_party1 = vectorizer.transform(df['first_party'])
    X_party2 = vectorizer.transform(df['second_party'])
    
    X = np.concatenate([X_party1.todense(), X_party2.todense(), X_facts.todense()], axis=1)
    return X
```
```Python
param_dist = {'learning_rate': uniform(0.01, 0.1),
              'max_depth': randint(3, 10),
              'n_estimators': randint(200, 500)}

random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=30, cv=3, random_state=42)

random_search.fit(X_train, Y_train)

best_model = random_search.best_estimator_
best_model
```
### WordNetLemmatizer + XGB
```Python
lemmatizer = WordNetLemmatizer()
def get_vector(vectorizer, df, train_mode):
    if train_mode:
        X_facts = vectorizer.fit_transform(df['facts'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split())))
    else:
        X_facts = vectorizer.transform(df['facts'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split())))
    X_party1 = vectorizer.transform(df['first_party'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split())))
    X_party2 = vectorizer.transform(df['second_party'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split())))
    
    X = np.concatenate([X_party1.todense(), X_party2.todense(), X_facts.todense()], axis=1)
    return X
```
```Python
model = xgb.XGBClassifier(learning_rate=0.1,
                          max_depth=3,
                          n_estimators=300)
```
### Word2Vec + XGB + Ensemble
```Python
word2vec_model = Word2Vec(sentences, vector_size=300, window=5, min_count=3, sg=1, epochs=30)
def get_word_embeddings(text):
    embeddings = []
    for word in text.split():
        if word in word2vec_model.wv:
            embeddings.append(word2vec_model.wv[word])
    if len(embeddings) > 0:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(word2vec_model.vector_size)
```
```Python
model1 = XGBClassifier(learning_rate=0.1, max_depth=5, n_estimators=100)
model2 = XGBClassifier(learning_rate=0.2, max_depth=3, n_estimators=200)
model3 = XGBClassifier(learning_rate=0.3, max_depth=7, n_estimators=150)

ensemble_model = VotingClassifier(
    estimators=[('model1', model1), ('model2', model2), ('model3', model3)],
    voting='hard')  # 분류 문제인 경우 'hard'로 설정
```
## 2. 성능 순위
1️⃣ TfidfVectorizer + LogisticRegression </br>
2️⃣ TfidfVectorizer + Word2Vec + XGB </br>
3️⃣ TfidfVectorizer + XGB </br>
4️⃣ Word2Vec + XGB + Ensemble </br>
5️⃣ WordNetLemmatizer + XGB
## 3. 회고
개인적으로 어렵다고 생각하던 자연어 처리 프로젝트를 진행해 보았습니다. 이 프로젝트를 통해 NLP에 대한 이해를 더욱 깊이 하게 되었으며, 다양한 종류의 NLP 모델들이 존재하며 NLP 분야는 특히 데이터 전처리, 모델 튜닝 등 많은 과정에서 많은 노력과 시간을 요구하는 것을 알게 되었습니다. 물론 프로젝트에서는 모든 종류의 모델을 직접 시도해보지는 못했지만, 특히 딥러닝 기술을 활용한 모델들이 높은 성능을 보인다는 점을 확인할 수 있었습니다. GPT, BART, BERT 등과 같은 모델들에 대해서도 공부해 볼 수 있는 기회를 가졌습니다. 프로젝트에서는 한정된 시간과 자원으로 더 많은 모델들을 시도하지 못한 점이 아쉽습니다.
