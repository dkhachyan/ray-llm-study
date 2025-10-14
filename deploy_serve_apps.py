import argparse
from typing import Optional

from ray.serve.schema import ServeDeploySchema
from ray.serve.scripts import _generate_config_from_file_or_import_path

from unverified_submission_client import UnverifiedServeSubmissionClient


def deploy(
    pre_delete: bool,
    config_path: str,
    client: UnverifiedServeSubmissionClient,
    name: Optional[str] = None,
):
    if pre_delete:
        client.delete_applications()
    args_dict = {}

    config: ServeDeploySchema = _generate_config_from_file_or_import_path(
        config_path,
        name=name,
        arguments=args_dict,
        runtime_env={},
    )
    for app in config.applications:
        app_runtime = app.runtime_env
        local_working_dir = app_runtime.get('local_working_dir')
        if local_working_dir is not None:
            app_runtime['working_dir'] = app_runtime.pop('local_working_dir')
            client._upload_working_dir_if_needed(app_runtime)

        for dep in app.deployments:
            dep_runtime = dep.ray_actor_options.runtime_env
            local_working_dir = dep_runtime.get('local_working_dir')
            if local_working_dir is not None:
                dep_runtime['working_dir'] = dep_runtime.pop('local_working_dir')
                client._upload_working_dir_if_needed(dep_runtime)
    print(config.applications)

    client.deploy_applications(
        config.dict(exclude_unset=True),
    )

    print(
        "\nSent deploy request successfully.\n "
        "* Use `serve status` to check applications' statuses.\n "
        "* Use `serve config` to see the current application config(s).\n"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--serve-config', type=str, required=True)
    parser.add_argument('--ray-dashboard-address', type=str, required=True)
    parser.add_argument('--ray-token-path', type=str, required=False, default='')
    parser.add_argument('--delete-all-apps-before-upload', action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ray_key=''
    if args.ray_token_path:
        with open(args.ray_token_path) as f:
            ray_key = f.read()

    serve_client = UnverifiedServeSubmissionClient(args.ray_dashboard_address,
                                                   headers=dict(Authorization=f"Bearer {ray_key}"))

    deploy(pre_delete=args.delete_all_apps_before_upload,
           config_path=args.serve_config,
           client=serve_client)
