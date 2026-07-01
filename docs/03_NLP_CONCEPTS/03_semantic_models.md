As the state of the art for NLP has advanced, the ability to train models that encapsulate the semantic relationship between tokens has led to the emergence of powerful deep learning language models. At the heart of these models is the encoding of language tokens as vectors (multi-valued arrays of numbers) known as embeddings.

This vector-based approach to modeling text became common with techniques like Word2Vec and GloVe, in which text tokens are represented as dense vectors with multiple dimensions. During model training, the dimension values are assigned to reflect semantic characteristics of each token based on their usage in the training text. The mathematical relationships between the vectors can then be exploited to perform common text analysis tasks more efficiently than older purely statistical techniques. A more recent advancement in this approach is to use a technique called attention to consider each token in context, and calculate the influence of the tokens around it. The resulting contextualized embeddings, such as those found in the GPT family of models, provide the basis of modern generative AI.


## Representing text as vectors
Vectors represent points in multidimensional space, defined by coordinates along multiple axes. Each vector describes a direction and distance from the origin. Semantically similar tokens should result in vectors that have a similar orientation – in other words they point in similar directions. The semantic characteristic encoded in the vectors makes it possible to use vector-based operations that compare words and enable analytical comparisons.

The vectors for "dog" and "cat" are similar (both domestic animals), as are "puppy" and "kitten" (both young animals). The words "tree", "young", and ball" have distinctly different vector orientations, reflecting their different semantic meanings.

The semantic characteristic encoded in the vectors makes it possible to use vector-based operations that compare words and enable analytical comparisons.


## Finding related terms
Since the orientation of vectors is determined by their dimension values, words with similar semantic meanings tend to have similar orientations. This means you can use calculations such as the cosine similarity between vectors to make meaningful comparisons.

For example, to determine the "odd one out" between "dog", "cat", and "tree", you can calculate the cosine similarity between pairs of vectors. The cosine similarity is calculated as:

cosine_similarity(A, B) = (A · B) / (||A|| * ||B||)

Where A · B is the dot product and ||A|| is the magnitude of vector A.

Example: "dog" and "cat" are highly similar (0.992), while "tree" has lower similarity to both "dog" (0.333) and "cat" (0.452). Therefore, tree is clearly the odd one out.


## Vector translation through addition and subtraction
You can add or subtract vectors to produce new vector-based results; which can then be used to find tokens with matching vectors. This technique enables intuitive arithmetic-based logic to determine appropriate terms based on linguistic relationships.

In practice, vector arithmetic rarely produces exact matches; instead, you would search for the word whose vector is closest (most similar) to the result.


## Analogical reasoning
Vector arithmetic can also answer analogy questions like "puppy is to dog as kitten is to ?"
Vector operations can capture linguistic relationships and enable reasoning about semantic patterns.


## Text summarization
Semantic embeddings enable extractive summarization by identifying sentences with vectors that are most representative of the overall document. By encoding each sentence as a vector (often by averaging or pooling the embeddings of its constituent words), you can calculate which sentences are most central to the document's meaning. These central sentences can be extracted to form a summary that captures the key themes.


## Keyword extraction
Vector similarity can identify the most important terms in a document by comparing each word's embedding to the document's overall semantic representation. Words whose vectors are most similar to the document vector, or most central when considering all word vectors in the document, are likely to be key terms that represent the main topics.


## Named entity recognition
Semantic models can be fine-tuned to recognize named entities (people, organizations, locations, etc.) by learning vector representations that cluster similar entity types together. During inference, the model examines each token's embedding and its context to determine whether it represents a named entity and, if so, what type.


## Text classification
For tasks like sentiment analysis or topic categorization, documents can be represented as aggregate vectors (such as the mean of all word embeddings in the document). These document vectors can then be used as features for machine learning classifiers, or compared directly to class prototype vectors to assign categories. Because semantically similar documents have similar vector orientations, this approach effectively groups related content and distinguishes different categories.