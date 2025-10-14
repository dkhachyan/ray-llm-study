from typing import Any, Dict, Optional

import requests
from ray._private.utils import split_address
from ray.dashboard.modules.dashboard_sdk import SubmissionClient
from ray.dashboard.modules.serve.sdk import ServeSubmissionClient

requests.packages.urllib3.disable_warnings()


class UnverifiedServeSubmissionClient(ServeSubmissionClient):
    def __init__(
            self,
            dashboard_head_address: str,
            create_cluster_if_needed=False,
            cookies: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            headers: Optional[Dict[str, Any]] = None,
    ):
        if requests is None:
            raise RuntimeError(
                "The Serve CLI requires the ray[default] "
                'installation: `pip install "ray[default]"`'
            )

        invalid_address_message = (
            "Got an unexpected address"
            f'"{dashboard_head_address}" while trying '
            "to connect to the Ray dashboard. The Serve SDK/CLI requires the "
            "Ray dashboard's HTTP(S) address (which should start with "
            '"http://" or "https://". If this address '
            "wasn't passed explicitly, it may be set in the "
            "RAY_DASHBOARD_ADDRESS environment variable."
        )

        if "://" not in dashboard_head_address:
            raise ValueError(invalid_address_message)

        module_string, _ = split_address(dashboard_head_address)

        # If user passes in ray://, raise error. Serve submission should
        # not use a Ray client address.
        if module_string not in ["http", "https"]:
            raise ValueError(invalid_address_message)

        SubmissionClient.__init__(self,
                                  address=dashboard_head_address,
                                  create_cluster_if_needed=create_cluster_if_needed,
                                  cookies=cookies,
                                  metadata=metadata,
                                  headers=headers,
                                  verify=False,
                                  )
        self._check_connection_and_version_with_url(
            min_version="1.12",
            version_error_message="Serve CLI is not supported on the Ray "
                                  "cluster. Please ensure the cluster is "
                                  "running Ray 1.12 or higher.",
            url="/api/ray/version",
        )
