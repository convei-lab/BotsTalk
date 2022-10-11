python scripts/self_mix.py \
--subtasks convai2,wizard_of_wikipedia,empatheticdialogues \
--num-self-mixs 5 \
--selfmix-max-turns 6 \
--datatype train \
--expert-model-files zoo:dodecadialogue/convai2_ft/model,zoo:dodecadialogue/wizard_of_wikipedia_ft/model,zoo:dodecadialogue/empathetic_dialogues_ft/model \
--expert-model-opt-files opt_files/conv.opt,opt_files/wow.opt,opt_files/ed.opt \
--display-examples True \
--task convai2 --seed_messages_from_task 1 \
--model-file zoo:dodecadialogue/convai2_ft/model \
--skip-generation False --inference nucleus \
--beam-size 3 \
--beam-min-length 10 --beam-block-ngram 3 --beam-context-block-ngram 3 \
--save-format parlai \
--ranker-model-files zoo:pretrained_transformers/model_poly/model,/home/minju/empathetic_dialogues_poly/model.checkpoint,/your_path/wizard_of_wikipedia_poly/model.checkpoint \
--outfile your_path/output/test_files.txt