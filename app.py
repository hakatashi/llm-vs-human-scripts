# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import signal
import sys
from types import FrameType

from flask import Flask

from utils.logging import logger
from comet import download_model, load_from_checkpoint

app = Flask(__name__)


def commet_score(commet_srcs, commet_mt, commet_ref):
    logger.info("Downloading COMET model")
    comet_model_path = download_model("Unbabel/wmt22-comet-da", local_files_only=True)
    comet_model = load_from_checkpoint(comet_model_path)
    comet_data = []
    for i in range(len(commet_srcs)):
        comet_instance = {}
        comet_instance["src"] = commet_srcs[i]
        comet_instance["mt"] = commet_mt[i]
        comet_instance["ref"] = commet_ref[i]
        comet_data.append(comet_instance)
    scores = comet_model.predict(comet_data, batch_size=8, gpus=1, progress_bar=False).scores
    return scores


@app.route("/")
def hello() -> str:
    # Use basic logging with custom fields
    logger.info(logField="custom-entry", arbitraryField="custom-entry")

    # https://cloud.google.com/run/docs/logging#correlate-logs
    logger.info("Child logger with trace Id.")

    src = [
        "The bodies of the five women who worked as sex workers in Ipswich were found around the town in December 2006.",
        "The bodies of the five women who worked as sex workers in Ipswich were found around the town in December 2006.",
        "The bodies of the five women who worked as sex workers in Ipswich were found around the town in December 2006.",
        "The bodies of the five women who worked as sex workers in Ipswich were found around the town in December 2006.",
        "The bodies of the five women who worked as sex workers in Ipswich were found around the town in December 2006.",
        "The bodies of the five women who worked as sex workers in Ipswich were found around the town in December 2006.",
    ]

    mt = [
        "2006年12月、イプスイッチでセックスワーカーとして働いていた女性五人が同町周辺で遺体となって発見された。",
        "イプスウィッチで性産業に従事していた5人の遺体が、2006年12月に町の周辺で発見された。",
        "イプスウィッチで性産業従事者として働いていた5人の女性は、2006年の12月に死体となって街の近くで発見された。",
        "イプスウィッチでセックスワーカーとして働いていた5人の女性の遺体が、2006年12月に町の周辺で発見された。",
        "イプスウィッチで性産業に従事していた五人の女性の遺体は、2006年12月に村周辺で見つかった。",
        "2006年12月に、イプスウィチで性産業に従事していた5人の女性の遺体が街の周辺で発見された。",
    ]

    ref = [
        "イプスウィッチで売春婦として働いていた5人の女性の遺体は、2006年12月に町の周辺で見つかった。",
        "イプスウィッチで売春婦として働いていた5人の女性の遺体は、2006年12月に町の周辺で見つかった。",
        "イプスウィッチで売春婦として働いていた5人の女性の遺体は、2006年12月に町の周辺で見つかった。",
        "イプスウィッチで売春婦として働いていた5人の女性の遺体は、2006年12月に町の周辺で見つかった。",
        "イプスウィッチで売春婦として働いていた5人の女性の遺体は、2006年12月に町の周辺で見つかった。",
        "イプスウィッチで売春婦として働いていた5人の女性の遺体は、2006年12月に町の周辺で見つかった。",
    ]

    score = commet_score(src, mt, ref)

    return f"COMET Score: {score}"


def shutdown_handler(signal_int: int, frame: FrameType) -> None:
    logger.info(f"Caught Signal {signal.strsignal(signal_int)}")

    from utils.logging import flush

    flush()

    # Safely exit program
    sys.exit(0)


if __name__ == "__main__":
    # Running application locally, outside of a Google Cloud Environment

    # handles Ctrl-C termination
    signal.signal(signal.SIGINT, shutdown_handler)

    app.run(host="localhost", port=8080, debug=True)
else:
    # handles Cloud Run container termination
    signal.signal(signal.SIGTERM, shutdown_handler)
