# TODO
-------------------------------
# NLP

## preprocessor
- fix Chunk size outliers - Some chunks are 38 chars (too small) or 19k chars (too big). theyve evaded filtering logic
- refactor:  should reuse jsonloader cache load/save/get() functionality

## jsonloader
- refactor: mv bert1 save/load_cache() + get_cache_path() to json_loader
	- reuse in BERT1 & RAG

## BERT1:
- try smaller chunks size
	- eg 400-600 (use overlap if splitting paragraphs to retain ctx) 🧪
	- rerun BERT1/prep-pipe with smaller chunksize ‼️
	- then run rag1 and see if less truncated chunks
- truncate chunks
	- add warning BERT1
	- truncate_to_max() sensible ? rm if possible
- load/save vector_db files in cache

## RAG:
- mk better use of bertscores
	- filter chunks (netzero or compound-score) ➡ reduces nChunks ➡ hope: speed⇧ & quality⇧
		- could reduce from 15k to 10k. not too aggressive to keep high recall
		- compound-score:  .4*spec + .4*commit + .2*commit + x*netZero --> what ratio?
	- add as metadata and process in rag OR in Topicmodel
	- **provenance** - link results to chunk_id and then use topicmodelling on these chunks
		- could use chunk_id to map bertscore metadata in TopicModel
- try batch_size >3 - observe speed/quality?
- try top_k >20
- with new embed (no-norm) tk Ti⇧ y❓
- eval: retr quali, embed quali
- rag output looks very similar, often only differing in 1 word. this is not picked up that well from topicmodellings embedding model, which puts them in similar semantic space, even though they might be quite different ➡ change rag output to me more diverse?

## LLM_EXTRACT
- ⚠️ groq: too many chunks hits limit fast.
	- optimize ctx_win
	- process per company or all at once. avoid extract_all being interrupted & have to restart all
- ⚠️ ollama: too slow
- pick correct model & adapt prompt
	- no hidden think like local qwen3
- if runs, compare results with rag. especially concerning recall@k

## model_test:
- eval metrics:
	- overlap
	- distribution across documents
	- are chunks diverse or repetitive?
- is test-format() actually needed for RAG? or does it need to be changed?
- would prompt need to be changed for RAG? or does it need to be changed?

## topic model:
- calc **coherence** metric using **gensim**
- add simple topic compare func to cmp b4/after param change
- TM output unique_company_count (how many companies report on this topic)
- how to preserve abbreviations❓ eg DRI / EAF

## global:
- modularity tests
	- run preprocess and then RAG2 without BERT1/RAG1
	- run prep-pipe wo translation and RAG multi-lingual chunks
	- run topicmodelling on only preprocessed/prep+bert chunks

## explain / reason / find solution
- think about new failure mode: Failure = retrieval failure OR extraction failure ? suggest solutions or explain
- explain difference when using exhaustive llm approach vs RAG

## results/
- Ausblick
- lessons learned
