(ns dk.cst.tf-idf
  "A reasonably performant TF-IDF implementation."
  (:require [clojure.string :as str]
            [clojure.java.math :as math]))

;; Should do a decent job with most text using English-like punctuation.
(def space+punctuation
  #"(\d|\s\p{Punct}+\s|\p{Punct}+\s|\s\p{Punct}+|\p{Punct}+$|\s)+")

(defn tokenize
  "Perform basic tokenization of `s`."
  [s]
  (str/split s space+punctuation))

(defn ->tokenizer-xf
  "Create a transducer for tokenizing documents based on various options:

    :preprocess  - fn for preprocessing a document.
    :tokenize    - fn taking a document as input, returning a coll of tokens.
    :postprocess - fn for postprocessing a coll of tokens.
    :ignored     - predicate determining which results to ignore."
  [& {:keys [preprocess tokenize postprocess ignored]
      :or   {tokenize    tokenize
             preprocess  str/lower-case
             postprocess identity
             ignored     #{[""]}}}]
  (comp
    (map preprocess)
    (map tokenize)
    (map postprocess)
    (remove ignored)
    (remove empty?)))

(def ^:dynamic *tokenizer-xf*
  "Default tokenizer transducer. Consider rebinding for other languages."
  (->tokenizer-xf))

(defn normalize-frequencies
  "Normalize the values of a `frequencies` map, dividing by `n`.

  If `n` is not provided, the sum of the frequencies is used in its place."
  ([frequencies n]
   (update-vals frequencies (fn [freq] (/ freq n))))
  ([frequencies]
   (normalize-frequencies frequencies (reduce + (vals frequencies)))))

(defn ->term-frequencies-xf
  "Create a transducer for calculating normalized frequency mappings of the
  tokens found in each document; initialize with a `tokenizer-xf`."
  [tokenizer-xf]
  (comp
    tokenizer-xf
    (map frequencies)
    (map normalize-frequencies)))

(defn ->document-terms-xf
  "Create a transducer for finding all unique tokens in each document;
  initialize with a `tokenizer-xf`"
  [tokenizer-xf]
  (comp
    tokenizer-xf
    (mapcat set)))

(defn tf
  "Normalized term frequencies for each of the `documents`."
  [documents]
  (into [] (->term-frequencies-xf *tokenizer-xf*) documents))

(defn list-terms
  "Concatenated list of unique terms found in each of the `documents`;
  will contain duplicates if multiple documents include the same term.

  If `tf-results` exist this may be used as input instead."
  [documents]
  (if (map? (first documents))                              ; tf results
    (mapcat keys documents)
    (into [] (->document-terms-xf *tokenizer-xf*) documents)))

(defn vocab
  "The vocabulary of the `documents`. If a `limit` is provided, then only return
  the vocabulary for terms with a per-document frequency higher than this limit.

  If a `df-result` or `idf-result` exists this may be used as input instead,
  but ONLY for the full vocabulary! For a limited vocabulary, substituting with
  `df-result` is OK, but `idf-result` is usually not UNLESS the `limit` is also
  similarly adjusted to account for the difference in values."
  ([documents]
   (if (map? documents)                                     ; df/idf result
     (into #{} (keys documents))
     (into #{} (->document-terms-xf *tokenizer-xf*) documents)))
  ([documents limit]
   (let [<=n? (comp #(<= % limit) second)
         xf   (comp (remove <=n?) (map first))]
     (if (map? documents)                                   ; df result
       (into #{} xf documents)
       (into #{} xf (frequencies (list-terms documents)))))))

(defn df
  "Per-document frequencies of each term in `documents` (not normalized).

  Input can be either a coll of strings or the output of 'tf'."
  [documents]
  (persistent!
    (reduce (fn [df-result term]
              (assoc! df-result term (inc (get df-result term 0))))
            (transient {})
            (list-terms documents))))

(defn invert
  "The inversion of `df-result` (normalized)."
  [df-result]
  (let [n (count df-result)]
    (update-vals df-result (fn [freq] (math/log (/ n (inc freq)))))))

(defn idf
  "Inverted frequencies of each term in `documents` (normalized)."
  [documents]
  (invert (df documents)))

(defn- apply-idf
  "Apply the `idf-result` to a `tf-result`; helper function for 'tf-idf'."
  [idf-result tf-result]
  (select-keys (merge-with * idf-result tf-result) (keys tf-result)))

(defn tf-idf
  "TF-IDF scores keyed to each term for all `documents`; will also return the
  TF-IDF scores for one `document` if a precomputed `idf-result` is provided."
  ([idf-result document]
   (apply-idf idf-result (first (tf [document]))))
  ([documents]
   (let [tf-results (tf documents)
         idf-result (idf tf-results)]
     (map (partial apply-idf idf-result) tf-results))))

(defn pick-terms
  "Pick terms in `tf-idf-results` according to a TF-IDF `score-picker`.

  The `score-picker` works as a reducing function for the scores of each term.
  A different `comparator` may be supplied to rank the results in another way."
  ([tf-idf-results score-picker comparator]
   (->> (apply merge-with score-picker tf-idf-results)
        (sort-by second comparator)
        (map first)))
  ([tf-idf-results score-picker]
   (pick-terms tf-idf-results score-picker >)))

(defn top-sum-terms
  "Top terms in `tf-idf-results` according to the sum TF-IDF score."
  [tf-idf-results]
  (pick-terms tf-idf-results +))

(defn top-max-terms
  "Top terms in `tf-idf-results` according to the max TF-IDF score."
  [tf-idf-results]
  (pick-terms tf-idf-results max))

(defn top-n-terms
  "Top `n` scoring terms for each of the `tf-idf-results`."
  [n tf-idf-results]
  (->> tf-idf-results
       (mapcat (comp
                 (partial map first)
                 (partial take n)
                 (partial sort-by second >)))
       (dedupe)))

(comment
  (def documents
    [""                                                     ; garbage data
     "...!"                                                 ; garbage data
     "Jeg har fri i dag."
     "Dagen i dag er en rigtig god dag."
     "Gode minder har vi heldigvis mange af."])

  ;; Core functions
  (tf documents)
  (df documents)
  (idf documents)
  (tf-idf documents)
  (tf-idf (idf documents) (nth documents 3))
  (vocab documents)
  (vocab (df documents))
  (vocab (idf documents))
  (vocab documents 1)
  (vocab (df documents) 1)

  ;; Utility functions
  (list-terms documents)
  (list-terms (tf documents))
  (normalize-frequencies (frequencies [:a :b :c :b :a]))
  (invert (df documents))

  ;; Tokenization
  (tokenize "Thomas' wife -- a true believer -- didn't think so.")
  (tokenize "Jeg har ønsket mig sådan en to-hovedet drage længe.")
  (into [] tokenization-xf documents)

  ;; Find top terms
  (take 6 (top-max-terms (tf-idf documents)))
  (take 6 (top-sum-terms (tf-idf documents)))
  (top-n-terms 2 (tf-idf documents))

  ;; Rebinding the tokenizer
  (binding [*tokenizer-xf* (->tokenizer-xf :tokenize #(str/split % #"a"))]
    (tf-idf documents))
  #_.)
