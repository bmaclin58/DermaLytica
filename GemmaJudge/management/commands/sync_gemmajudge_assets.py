from django.core.management.base import BaseCommand

from GemmaJudge.services.runtime import get_asset_sync_service


class Command(BaseCommand):
    help = "Sync the Gemma Judge Chroma and embedding-model assets from GCS to the local runtime cache."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force",
            action="store_true",
            help="Delete any existing local cache and re-sync from GCS.",
        )

    def handle(self, *args, **options):
        synced = get_asset_sync_service().ensure_assets(force=options["force"])
        if synced:
            for path in synced:
                self.stdout.write(self.style.SUCCESS(f"Synced asset cache: {path}"))
        else:
            self.stdout.write(self.style.SUCCESS("Asset cache already up to date."))
