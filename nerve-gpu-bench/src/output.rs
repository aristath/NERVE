use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Publishes a complete artifact with a same-directory rename. A failed write
/// never truncates the previously accepted result.
pub fn write_atomic(path: &Path, payload: &[u8]) -> Result<(), io::Error> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent)?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("benchmark-output"),
        std::process::id(),
        nonce,
    ));
    let write_result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(payload)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        drop(file);
        fs::rename(&temporary, path)?;
        OpenOptions::new().read(true).open(parent)?.sync_all()
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn temporary_test_directory(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "nerve-gpu-bench-{name}-{}-{nonce}",
            std::process::id()
        ))
    }

    #[test]
    fn atomic_write_replaces_only_the_requested_output() {
        let directory = temporary_test_directory("atomic-output");
        fs::create_dir_all(&directory).unwrap();
        let output = directory.join("catalog.json");
        fs::write(&output, b"stale").unwrap();

        write_atomic(&output, b"{\"schema\":\"test\"}").unwrap();

        assert_eq!(fs::read(&output).unwrap(), b"{\"schema\":\"test\"}\n");
        assert_eq!(fs::read_dir(&directory).unwrap().count(), 1);
        fs::remove_dir_all(directory).unwrap();
    }
}
