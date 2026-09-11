 if (
            top.in_both_retrievers
            and unique_sources >= 2
        ):
            confidence = "strong"
        elif top.in_both_retrievers:
            confidence = "moderate"
        else:
            confidence = "weak"

        return final, Evidence(
            level=confidence,

            should_answer=True,

            top_score=top.rrf_score,

            score_source="rrf",

            supporting_chunks=len(final),

            top_gap=None,
            
            agreement=top.in_both_retrievers,
            top_percentile=None,
            retrieval_mode=retrieval_mode,
            reason=(
                "Answer generated from rank-based "
                "retrieval because CrossEncoder "
                "reranking was unavailable."),
        )

